from openai import OpenAI
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
from termcolor import colored
import tiktoken

load_dotenv()

client = OpenAI()
GPT_MODEL = "gpt-3.5-turbo-16k"


# web scraping
def scrape(url):
    app = FirecrawlApp()

    # scrape a single URL

    try:
        scraped_data = app.scrape_url(url)
    except Exception as e:
        print("Unable to scape the url ")
        print(f"Exception: {e}")
        return e

    links_scraped.append(url)

    return scraped_data["markdown"]


def search(query, enity_name: str):
    app = FirecrawlApp()
    params = {"pageOptions": {"fetchPageContent": True}}

    # scrape a single URL
    serch_result = app.search(query, params=params)
    search_result_str = str(serch_result)

    data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]

    prompt = f""" {search_result_str} 
    -------

    About is some search results from the internet about {query}
    Your goal is to find a specific list of information called { enity_name } regarding {data_keys_to_search}

    Please extract information from the search results above in specific JSON format:

    {{
        "related files to scrape further": ["url1", "url2", "url3"],
        'info found': {{
            "research_item_ 1": 'xxxx',
            "reference": url
        }},
        {{
            "research_item_ 2": 'xxxx',
            "reference": url
        }},
        ...]
    }}

    Where research_item_1, research_item_2, ... are the actual research item names you are looking for;
    Only return research_items that you actually found in the search results above
    If no research item information is found from the content provided, just dont return anything for that research item

    Extracted JSON:
    """

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    result = response.choices[0].message.content

    return result


def update_data(datas_update):
    """
    Update the state with the new data points found

    Args:
        state (dict): the current graph state
        datas_update (List[dict]): the new data points found, have to follow the format[{"name": "xxx", "value": "xxx", "reference": "xxx"}]

    Returns:
        state (dict): the updated graph state
    """
    print(f"Updating data points: {datas_update}")

    for data in datas_update:
        for obj in data_points:
            if obj["name"] == data["name"]:
                obj["value"] = data["value"]
                obj["reference"] = data["reference"]

    return f"data updated: {data_points}"


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_compleation_request(messages, tool_choice, tools, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model, tool_choice=tool_choice, tools=tools, messages=messages
        )
        return response.choices[0].message.content

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversion(message):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    if message["role"] == "system":
        print(colored(f"System: {message['content']}", role_to_color[message["role"]]))
    elif message["role"] == "user":
        print(colored(f"User: {message['content']}", role_to_color[message["role"]]))
    elif message["role"] == "assistant" and message.get("tool"):
        print(
            colored(
                f"Assistant: {message['tool_calls']}\n",
                role_to_color[message["role"]],
            )
        )
    elif message["role"] == "assistant" and not message.get("tool_calls"):
        print(
            colored(
                f"Assistant: {message['content']}\n",
                role_to_color[message["role"]],
            )
        )


tools_list = {"scrape": scrape, "search": search, "update_data": update_data}


def memory_optimise(messages: list):
    system_prompt = messages[0]["content"]

    # token count
    encoding = tiktoken.encoding_for_model(GPT_MODEL)

    if len(messages) > 24 or len(encoding.encode(str())) > 10000:
        latest_messages = messages[-12:]

        token_count_latest_messages = len(encoding.encode(str(latest_messages)))
        print(f"Token count for latest messages: {token_count_latest_messages}")

        index = messages.index(latest_messages[0])
        early_messages = messages[:index]

        prompt = f"""{early_messages}
        ------
        Above is the past history of conversation between the user and the assistant, including actions the assistant took
        Give a summary of the past actions taken so far, key information, and compleated tasks

        SUMMARY:
        """

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        system_prompt = f"""{system_prompt}; Here is a summary of past actions taken so far: {response[0]['generated_text']}"""
        messages = [{"role": "system", "content": system_prompt}] + latest_messages

        return messages

    return messages


def call_agent(prompt, system_prompt, tools, plan):
    messages = []

    if plan:
        messages.append(
            {
                "role": "user",
                "contrnt": (
                    system_prompt
                    + " "
                    + prompt
                    + " Lets think step by step, make a plan first"
                ),
            }
        )

        print(messages)
        chat_response = chat_compleation_request(
            messages, tool_choice="none", tools=tools
        )
        print(chat_response.choices[0].message.content)
        messages = [
            {"role": "user", "content": (system_prompt + " " + prompt)},
            {"role": "assistant", "content": chat_response.choices[0].message.content},
        ]

    else:
        messages.append({"role": "user", "content": (system_prompt + " " + prompt)})

    state = "running"

    for message in messages:
        pretty_print_conversion(message)

    while state == "running":
        chat_response = chat_compleation_request(
            messages, tool_choice=None, tools=tools
        )

        if isinstance(chat_response, Exception):
            print("Failed to get a valid response: ", chat_response)
            state = "finished"
        else:
            current_choice = chat_response.choices[0]
            messages.append(
                {
                    "role": "assistant",
                    "content": current_choice.message.content,
                    "tool_calls": current_choice.tool_calls,
                }
            )

            pretty_print_conversion(messages[-1])

            if "tool_calls" in current_choice:
                tool_calls = current_choice["tool_calls"]

                for tool_call in tool_calls:
                    function = tool_call["function"]
                    parameters = tool_call["parameters"]

                    if function not in tools_list:
                        print(f"Function {function} not found in tools list")
                        continue

                    result = tools_list[function](**parameters)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function,
                        "content": result,
                    }
                )
            pretty_print_conversion(messages[-1])

        if (
            "finish_reason" in current_choice
            and current_choice["finish_reason"] == "stop"
        ):
            state = "finished"
        messages = memory_optimise(messages)
    return messages[-1]["content"]


# step 1: run agent to do website search
def website_search(entity_name: str, website: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scrape",
                "description": "Scrape a website for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_data",
                "description": "Save data points found for later use",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "datas_update": {
                            "type": "array",
                            "description": "The data points to update",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the data point",
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "The value of the data point",
                                    },
                                    "reference": {
                                        "type": "string",
                                        "description": "The reference for the data point",
                                    },
                                },
                            },
                        }
                    },
                    "required": ["datas_update"],
                },
            },
        },
    ]

    data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]

    system_prompt = """
    You are a world class webscraper, you are great at finding information on the internet using urls;
    You will keep scraping a url based on information you recieved until information is found;
    If you cannot find the relevent information from the compan's domain related urls,
    Whenever you find certain data points, you will  use "update_data" function to save the data point;
    You only answer questions on results from scraper do not make anything up;
    You NEVER ask user for inputs of permissions,
    Just go ahead and do the best things possible without asking for permission or guidance
    """

    prompt = f"""
    Entity to search: {entity_name}

    Company website: {website}

    Data points to search: {data_keys_to_search}
    """

    response = call_agent(prompt, system_prompt, tools, plan=True)
    return response


# step 2: run agent to do internet search
def internet_search(entity_name: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scrape",
                "description": "Scrape a url for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_data",
                "description": "Save data points found for later use",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "datas_update": {
                            "type": "array",
                            "description": "The data points to update",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the data point",
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "The value of the data point",
                                    },
                                    "reference": {
                                        "type": "string",
                                        "description": "The reference for the data point",
                                    },
                                },
                            },
                        }
                    },
                    "required": ["datas_update"],
                },
            },
        },
    ]

    data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]

    if len(data_keys_to_search > 0):
        system_prompt = """
        You are a world class web researcher
        You will keep scraping a url based on information you recieved until information is found;

        You will try as hard as possible to find all sorts of query & source to find information on the internet
        You will keep searching until you find the information you are looking for;
        Whenever you find certain data points, you will  use "update_data" function to save the data point;
        You only answer questions on results from scraper do not make anything up;

        You NEVER ask user for inputs of permissions just go ahead and do the best things possible without asking for permission or guidance
        You ONLY run one function at a time Never run multiple functions at the same time
        """

        prompt = f"""
        Entity to search: {entity_name}

        Links we already scraped: {links_scraped}

        Data points to search: 
        {data_keys_to_search}
        """

        response = call_agent(prompt, system_prompt, tools, plan=True)
        return response


links_scraped = []

data_points = [
    {"name": "function_name", "value": None, "reference": None},
    {"name": "function_code", "value": None, "reference": None},
    {"name": "code_language", "value": None, "reference": None},
]

entity_name = "Github"
website = "https://github.com"

responseOne = website_search(entity_name, website)
responseTwo = internet_search(entity_name)

print("------")
print(f"Data points found: {data_points}")
