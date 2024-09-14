"""
This script is adapted from work found at:
https://github.com/tiny-rawr/parallel_process_gpt,
which itself is based on examples provided by OpenAI at:
https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

MIT License

Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
from typing import List  # for type hints in functions
from pydantic import BaseModel, Field

class OAIApiFromFileConfig(BaseModel):
 requests_filepath: str
 save_filepath: str
 api_key: str
 request_url:str =  Field("https://api.openai.com/v1/embeddings",description="The url to use for generating embeddings")
 max_requests_per_minute: float = Field(100,description="The maximum number of requests per minute")
 max_tokens_per_minute: float = Field(1_000_000,description="The maximum number of tokens per minute")
 max_attempts:int = Field(5,description="The maximum number of attempts to make for each request")
 logging_level:int = Field(20,description="The logging level to use for the request")
 token_encoding_name: str = Field("cl100k_base",description="The token encoding scheme to use for calculating request sizes")

async def process_api_requests_from_file(
        api_cfg: OAIApiFromFileConfig
):
    """
    Asynchronously processes API requests from a given file, executing them in parallel
    while adhering to specified rate limits for requests and tokens per minute.
    
    This function reads a file containing JSONL-formatted API requests, sends these requests
    concurrently to the specified API endpoint, and handles retries for failed attempts,
    all the while ensuring that the execution does not exceed the given rate limits.
    
    Parameters:
    - requests_filepath: Path to the file containing the JSONL-formatted API requests.
    - save_filepath: Path to the file where results or logs should be saved.
    - request_url: The API endpoint URL to which the requests will be sent.
    - api_key: The API key for authenticating requests to the endpoint.
    - max_requests_per_minute: The maximum number of requests allowed per minute.
    - max_tokens_per_minute: The maximum number of tokens (for rate-limited APIs) that can be used per minute.
    - token_encoding_name: Name of the token encoding scheme used for calculating request sizes.
    - max_attempts: The maximum number of attempts for each request in case of failures.
    - logging_level: The logging level to use for reporting the process's progress and issues.
    
    The function initializes necessary tracking structures, sets up asynchronous HTTP sessions,
    and manages request retries and rate limiting. It logs the progress and any issues encountered
    during the process to facilitate monitoring and debugging.
    """
    #extract variables from config
    requests_filepath = api_cfg.requests_filepath
    save_filepath = api_cfg.save_filepath
    request_url = api_cfg.request_url
    api_key = api_cfg.api_key
    max_requests_per_minute = api_cfg.max_requests_per_minute
    max_tokens_per_minute = api_cfg.max_tokens_per_minute
    token_encoding_name = api_cfg.token_encoding_name
    max_attempts = api_cfg.max_attempts
    logging_level = api_cfg.logging_level
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if '/deployments' in request_url:
        request_header = {"api-key": f"{api_key}"}
    # Add Anthropic-specific headers
    if 'anthropic.com' in request_url:
        request_header = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "prompt-caching-2024-07-31"
        }

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            metadata, actual_request = request_json  # Unpack the list
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=actual_request,
                                token_consumption=num_tokens_consumed_from_request(
                                    actual_request, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=metadata,
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses


@dataclass
class StatusTracker:
    """
    A data class that tracks the progress and status of API request processing.
    
    This class is designed to hold counters for various outcomes of API requests
    (such as successes, failures, and specific types of errors) and other relevant
    metadata to manage and monitor the execution flow of the script.
    
    Attributes:
    - num_tasks_started: The total number of tasks that have been started.
    - num_tasks_in_progress: The current number of tasks that are in progress.
      The script continues running as long as this number is greater than 0.
    - num_tasks_succeeded: The total number of tasks that have completed successfully.
    - num_tasks_failed: The total number of tasks that have failed.
    - num_rate_limit_errors: The count of errors received due to hitting the API's rate limits.
    - num_api_errors: The count of API-related errors excluding rate limit errors.
    - num_other_errors: The count of errors that are neither API errors nor rate limit errors.
    - time_of_last_rate_limit_error: A timestamp (as an integer) of the last time a rate limit error was encountered,
      used to implement a cooling-off period before making subsequent requests.
    
    The class is initialized with all counters set to 0, and the `time_of_last_rate_limit_error`
    set to 0 indicating no rate limit errors have occurred yet.
    """

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """
    Represents an individual API request with associated metadata and the capability to asynchronously call an API.
    
    Attributes:
    - task_id (int): A unique identifier for the task.
    - request_json (dict): The JSON payload to be sent with the request.
    - token_consumption (int): Estimated number of tokens consumed by the request, used for rate limiting.
    - attempts_left (int): The number of retries left if the request fails.
    - metadata (dict): Additional metadata associated with the request.
    - result (list): A list to store the results or errors from the API call.
    
    This class encapsulates the data and actions related to making an API request, including
    retry logic and error handling.
    """

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """
        Asynchronously sends the API request using aiohttp, handles errors, and manages retries.
        
        Parameters:
        - session (aiohttp.ClientSession): The session object used for HTTP requests.
        - request_url (str): The URL to which the request is sent.
        - request_header (dict): Headers for the request, including authorization.
        - retry_queue (asyncio.Queue): A queue for requests that need to be retried.
        - save_filepath (str): The file path where results or errors should be logged.
        - status_tracker (StatusTracker): A shared object for tracking the status of all API requests.
        
        This method attempts to post the request to the given URL. If the request encounters an error,
        it determines whether to retry based on the remaining attempts and updates the status tracker
        accordingly. Successful requests or final failures are logged to the specified file.
        """
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                self.metadata["end_time"] = time.time()
                self.metadata["total_time"] = self.metadata["end_time"] - self.metadata["start_time"]
                data = [self.metadata, self.request_json, {"error": str(error)}]
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            self.metadata["end_time"] = time.time()
            self.metadata["total_time"] = self.metadata["end_time"] - self.metadata["start_time"]
            data = [self.metadata, self.request_json, response]
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url: str) -> str:
    """
    Extracts the API endpoint from a given request URL.

    This function applies a regular expression search to find the API endpoint pattern within the provided URL.
    It supports extracting endpoints from standard OpenAI API URLs as well as custom Azure OpenAI deployment URLs.

    Parameters:
    - request_url (str): The full URL of the API request.

    Returns:
    - str: The extracted endpoint from the URL. If the URL does not match expected patterns,
      this function may raise an IndexError for accessing a non-existing match group.

    Example:
    - Input: "https://api.openai.com/v1/completions"
      Output: "completions"
    - Input: "https://custom.azurewebsites.net/openai/deployments/my-model/completions"
      Output: "completions"
    """
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
        if match is None:
            raise ValueError(f"Invalid URL: {request_url}")
    return match[1] 


def append_to_jsonl(data, filename: str) -> None:
    """
    Appends a given JSON payload to the end of a JSON Lines (.jsonl) file.

    Parameters:
    - data: The JSON-serializable Python object (e.g., dict, list) to be appended.
    - filename (str): The path to the .jsonl file to which the data will be appended.

    The function converts `data` into a JSON string and appends it to the specified file,
    ensuring that each entry is on a new line, consistent with the JSON Lines format.

    Note:
    - If the specified file does not exist, it will be created.
    - This function does not return any value.
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Supports completion, embedding, and Anthropic message requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    elif api_endpoint == "messages":  # Anthropic API
        messages = request_json.get("messages", [])
        num_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                num_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        num_tokens += len(encoding.encode(item["text"]))
        
        max_tokens = request_json.get("max_tokens", 0)
        num_tokens += max_tokens  # Add the max_tokens to account for the response
        return num_tokens
    
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """
    Generates a sequence of integer task IDs, starting from 0 and incrementing by 1 each time.

    Yields:
    - An integer representing the next task ID in the sequence.

    This generator function is useful for assigning unique identifiers to tasks or requests
    in a sequence, ensuring each has a distinct ID for tracking and reference purposes.
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    config = OAIApiFromFileConfig(
        requests_filepath=args.requests_filepath,
        save_filepath=args.save_filepath,
        request_url=args.request_url,
        api_key=args.api_key,
        max_requests_per_minute=float(args.max_requests_per_minute),
        max_tokens_per_minute=float(args.max_tokens_per_minute),
        token_encoding_name=args.token_encoding_name,
        max_attempts=int(args.max_attempts),
        logging_level=int(args.logging_level),
    )
    asyncio.run(process_api_requests_from_file(api_cfg=config))