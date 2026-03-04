from mcp_agent import MCPAgent


def main():

    agent = MCPAgent("CosmicAgent")

    print("\nAgent ready. Type 'exit' to quit.\n")

    while True:

        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            break

        reply = agent.step(user)

        print("\nAgent:", reply, "\n")

    agent.export_memory()
    agent.shutdown()


if __name__ == "__main__":
    main()