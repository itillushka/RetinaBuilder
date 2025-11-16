"""
Parallel Execution Manager

Manages parallel processing of volume alignment tasks in multi-volume panoramic stitching.

Features:
- Concurrent execution of independent alignment tasks
- Process pool management
- Error handling and recovery
- Progress tracking
- Results collection

Uses multiprocessing for true parallelism (not limited by Python GIL).

Author: OCT Panoramic Stitching System
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Callable, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AlignmentTask:
    """Represents a single volume alignment task."""

    reference_id: int
    moving_id: int
    reference_volume: Any  # np.ndarray (can't serialize type hint easily)
    moving_volume: Any     # np.ndarray
    level: int
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"AlignmentTask(ref={self.reference_id}, mov={self.moving_id}, level={self.level})"


@dataclass
class AlignmentResult:
    """Stores result of a volume alignment task."""

    reference_id: int
    moving_id: int
    level: int
    success: bool

    # Alignment results (if successful)
    transform: Optional[Any] = None  # Transform3D object
    aligned_volume: Optional[Any] = None  # np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)

    # Error information (if failed)
    error: Optional[str] = None
    traceback: Optional[str] = None

    # Timing
    duration: float = 0.0

    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"AlignmentResult({self.reference_id}←{self.moving_id}, {status}, {self.duration:.1f}s)"


class ParallelAlignmentExecutor:
    """
    Manages parallel execution of alignment tasks.

    Uses process pool for CPU-bound alignment operations.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker processes
                        (default: min(4, cpu_count()))
        """
        if max_workers is None:
            # Conservative default: use up to 4 cores
            max_workers = min(4, mp.cpu_count())

        self.max_workers = max_workers
        logger.info(f"Parallel executor initialized with {max_workers} workers")

    def execute_level(self,
                     tasks: List[AlignmentTask],
                     alignment_func: Callable,
                     show_progress: bool = True) -> List[AlignmentResult]:
        """
        Execute all alignment tasks for a processing level in parallel.

        Args:
            tasks: List of alignment tasks to execute
            alignment_func: Function to align a single pair
                           Signature: (ref_vol, mov_vol, params) -> (transform, aligned_vol, metrics)
            show_progress: Whether to show progress updates

        Returns:
            List of AlignmentResult objects
        """
        if not tasks:
            logger.warning("No tasks to execute")
            return []

        level = tasks[0].level
        logger.info(f"\nExecuting Level {level}: {len(tasks)} alignment tasks in parallel")

        results = []

        # Execute tasks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                future = executor.submit(
                    self._execute_single_task,
                    task,
                    alignment_func
                )
                future_to_task[future] = task

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        logger.info(f"  [{completed}/{len(tasks)}] ✓ {task} completed in {result.duration:.1f}s")
                    else:
                        logger.error(f"  [{completed}/{len(tasks)}] ✗ {task} FAILED: {result.error}")

                except Exception as e:
                    logger.error(f"  [{completed}/{len(tasks)}] ✗ {task} EXCEPTION: {e}")
                    results.append(AlignmentResult(
                        reference_id=task.reference_id,
                        moving_id=task.moving_id,
                        level=task.level,
                        success=False,
                        error=str(e)
                    ))

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.duration for r in results)

        logger.info(f"\nLevel {level} complete: {successful}/{len(tasks)} successful, {failed} failed")
        logger.info(f"Total time: {total_time:.1f}s (parallel), {total_time/self.max_workers:.1f}s (equivalent sequential)")

        return results

    @staticmethod
    def _execute_single_task(task: AlignmentTask,
                            alignment_func: Callable) -> AlignmentResult:
        """
        Execute a single alignment task.

        This is a static method to ensure it's picklable for multiprocessing.

        Args:
            task: Alignment task to execute
            alignment_func: Alignment function

        Returns:
            AlignmentResult object
        """
        start_time = time.time()

        try:
            # Run alignment
            transform, aligned_volume, metrics = alignment_func(
                task.reference_volume,
                task.moving_volume,
                task.params
            )

            duration = time.time() - start_time

            return AlignmentResult(
                reference_id=task.reference_id,
                moving_id=task.moving_id,
                level=task.level,
                success=True,
                transform=transform,
                aligned_volume=aligned_volume,
                metrics=metrics,
                duration=duration
            )

        except Exception as e:
            import traceback as tb
            duration = time.time() - start_time

            return AlignmentResult(
                reference_id=task.reference_id,
                moving_id=task.moving_id,
                level=task.level,
                success=False,
                error=str(e),
                traceback=tb.format_exc(),
                duration=duration
            )


class SequentialExecutor:
    """
    Sequential execution of alignment tasks (for debugging or single-threaded mode).

    Provides same interface as ParallelAlignmentExecutor but executes sequentially.
    """

    def __init__(self):
        logger.info("Sequential executor initialized (single-threaded mode)")

    def execute_level(self,
                     tasks: List[AlignmentTask],
                     alignment_func: Callable,
                     show_progress: bool = True) -> List[AlignmentResult]:
        """
        Execute all alignment tasks sequentially.

        Args:
            tasks: List of alignment tasks
            alignment_func: Alignment function
            show_progress: Whether to show progress

        Returns:
            List of AlignmentResult objects
        """
        if not tasks:
            return []

        level = tasks[0].level
        logger.info(f"\nExecuting Level {level}: {len(tasks)} alignment tasks (sequential)")

        results = []

        for i, task in enumerate(tasks):
            logger.info(f"  [{i+1}/{len(tasks)}] Processing {task}...")

            result = ParallelAlignmentExecutor._execute_single_task(task, alignment_func)
            results.append(result)

            if result.success:
                logger.info(f"    ✓ Completed in {result.duration:.1f}s")
            else:
                logger.error(f"    ✗ FAILED: {result.error}")

        # Summary
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.duration for r in results)

        logger.info(f"\nLevel {level} complete: {successful}/{len(tasks)} successful")
        logger.info(f"Total time: {total_time:.1f}s")

        return results


def create_executor(parallel: bool = True,
                   max_workers: Optional[int] = None):
    """
    Factory function to create an executor.

    Args:
        parallel: If True, use parallel executor; if False, use sequential
        max_workers: Maximum worker processes (for parallel executor)

    Returns:
        Executor instance
    """
    if parallel:
        return ParallelAlignmentExecutor(max_workers=max_workers)
    else:
        return SequentialExecutor()


def dummy_alignment_func(ref_vol, mov_vol, params):
    """
    Dummy alignment function for testing.

    Args:
        ref_vol: Reference volume
        mov_vol: Moving volume
        params: Parameters dict

    Returns:
        (transform, aligned_volume, metrics)
    """
    import numpy as np
    time.sleep(1)  # Simulate work

    # Create dummy transform
    from .coordinate_system import Transform3D
    transform = Transform3D(
        dx=10.0, dy=5.0, dz=0.0,
        rotation_z=2.0,
        confidence=0.85
    )

    # Return moving volume as-is (no actual alignment)
    metrics = {
        'ncc_before': 0.70,
        'ncc_after': 0.88,
        'execution_time': 1.0
    }

    return transform, mov_vol, metrics


def main():
    """Test the parallel executor."""
    import numpy as np
    from .coordinate_system import Transform3D

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create dummy volumes
    dummy_vol = np.random.rand(100, 200, 50).astype(np.float32)

    # Create tasks
    tasks = [
        AlignmentTask(1, 2, dummy_vol, dummy_vol, level=1),
        AlignmentTask(1, 4, dummy_vol, dummy_vol, level=1),
        AlignmentTask(1, 6, dummy_vol, dummy_vol, level=1),
        AlignmentTask(1, 8, dummy_vol, dummy_vol, level=1),
    ]

    # Test parallel execution
    print("\n" + "="*70)
    print("TESTING PARALLEL EXECUTION")
    print("="*70)

    executor = ParallelAlignmentExecutor(max_workers=2)
    results = executor.execute_level(tasks, dummy_alignment_func)

    print("\nResults:")
    for result in results:
        print(f"  {result}")

    # Test sequential execution
    print("\n" + "="*70)
    print("TESTING SEQUENTIAL EXECUTION")
    print("="*70)

    executor_seq = SequentialExecutor()
    results_seq = executor_seq.execute_level(tasks, dummy_alignment_func)

    print("\nResults:")
    for result in results_seq:
        print(f"  {result}")


if __name__ == "__main__":
    main()
