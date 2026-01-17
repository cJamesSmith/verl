import asyncio
from typing import List, Tuple

from sandbox_fusion import RunCodeRequest, RunCodeResponse, RunStatus, run_code_async, set_sandbox_endpoint


async def single_sandbox(code: str, stdin: str = "", language: str = "python", compile_timeout: int = 10, run_timeout: float = 3.0, semaphore: asyncio.Semaphore = None) -> RunCodeResponse:
    request = RunCodeRequest(
        code=code,
        stdin=stdin,
        language=language,
        compile_timeout=compile_timeout,
        run_timeout=run_timeout,
    )
    async with semaphore:
        response = await run_code_async(request, client_timeout=30.0, max_attempts=2)
        response = response.dict()
    await asyncio.sleep(2)
    return response


async def parallel_sandbox(tasks: list, stdin_list: list = None, num_processes: int = 200) -> Tuple[List[bool], List[str], List[str]]:
    semaphore = asyncio.Semaphore(num_processes)
    set_sandbox_endpoint("http://localhost:8080/")
    if stdin_list is None:
        tasks_async = [single_sandbox(code=code, semaphore=semaphore) for code in tasks]
    else:
        assert len(tasks) == len(stdin_list), f"len(tasks) (f{len(tasks)}) != len(stdin_list) ({len(stdin_list)})"
        tasks_async = [single_sandbox(code=code, stdin=stdin, semaphore=semaphore) for code, stdin in zip(tasks, stdin_list)]
    results = await asyncio.gather(*tasks_async, return_exceptions=False)
    return (
        [r["status"] == RunStatus.Success for r in results],
        [r["run_result"]["stdout"] for r in results],
        [r["run_result"]["stderr"] for r in results],
    )


if __name__ == "__main__":
    code_list = [
        """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
""",
        """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
""",
        """
name = input("Your name:")
print(f"Hi, {name}!")
""",
    ]
    stdin_list = ["", "", "Alice"]

    for code, stdin in zip(code_list, stdin_list):
        print(f"code: {code}")
        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(parallel_sandbox(tasks=[code], stdin_list=[stdin]))

        print(f"sandbox_success: {sandbox_success}")
        print(f"sandbox_stdout: {sandbox_stdout}")
        print(f"sandbox_stderr: {sandbox_stderr} {sandbox_stderr[0].splitlines()[-1:]}")
