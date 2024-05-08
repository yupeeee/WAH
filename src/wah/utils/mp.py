from multiprocessing import Manager, Process, Queue
from queue import Empty
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
)

__all__ = [
    "MPManager",
]


def worker(
    tasks: Queue,
    results: Queue,
    work: Callable,
    work_args: Optional[Tuple] = (),
) -> None:
    while True:
        try:
            task = tasks.get_nowait()
            result = work(task, *work_args)
            results.put((task, result))

        except Empty:
            break


class MPManager:
    def __init__(
        self,
        nprocs: int,
        tasks: List[Any],
    ) -> None:
        self.nprocs: int = nprocs
        self.tasks: Queue = Manager().Queue()
        self.results: Queue = Manager().Queue()

        for task in tasks:
            self.tasks.put(task)

    def run(
        self,
        work: Callable,
        work_args: Optional[Tuple] = (),
    ) -> List[Tuple[int, Any]]:
        procs: List[Process] = []

        for _ in range(self.nprocs):
            proc = Process(
                target=worker, args=(self.tasks, self.results, work, work_args)
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        results = []
        while not self.results.empty():
            results.append(self.results.get())

        return results
