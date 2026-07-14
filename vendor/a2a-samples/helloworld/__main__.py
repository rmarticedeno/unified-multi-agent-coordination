import uvicorn

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from agent_executor import HelloWorldAgentExecutor
from starlette.applications import Starlette


if __name__ == '__main__':
    skill = AgentSkill(
        id='echo_bot',
        name='Echo Bot',
        description=(
            'An example agent that acknowledges client request and responds with a '
            '"Hello World" message.'
        ),
        input_modes=['text/plain'],
        output_modes=['text/plain'],
        tags=['a2a', 'echo-example'],
        examples=['hi', 'how are you'],
    )
    extended_skill = AgentSkill(
        id='echo_bot_super_mode',
        name='Echo Bot (Super Mode)',
        description='An extended version of Echo Bot that responds with extra enthusiasm!',
        tags=['a2a', 'echo-example', 'extended'],
        examples=['super hi', 'give me a super hello'],
    )
    public_agent_card = AgentCard(
        name='Hello World Agent',
        description='Just a hello world agent',
        version='0.0.1',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9999',
                protocol_version='1.0',
            )
        ],
        skills=[skill],
    )
    extended_agent_card = AgentCard(
        name='Hello World Agent - Extended Edition',
        description='The full-featured hello world agent for authenticated users.',
        version='0.0.2',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9999',
                protocol_version='1.0',
            )
        ],
        skills=[skill, extended_skill],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=public_agent_card,
        extended_agent_card=extended_agent_card,
    )
    routes = [
        *create_agent_card_routes(public_agent_card),
        *create_jsonrpc_routes(request_handler, '/'),
    ]
    app = Starlette(routes=routes)
    uvicorn.run(app, host='127.0.0.1', port=9999)
