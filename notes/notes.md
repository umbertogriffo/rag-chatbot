# Notes

From [François Chollet](https://arcprize.org/blog/oai-o3-pub-breakthrough) a nice definition of what is an `LLM`:

> We can see LLMs working as a [repository of vector programs](https://fchollet.substack.com/p/how-i-think-about-llm-prompt-engineering).
When prompted, they will fetch the program that your prompt maps to and "execute" it on the input at hand.
LLMs are a way to store and operationalize millions of useful mini-programs via passive exposure to human-generated content.
This "memorize, fetch, apply" paradigm can achieve arbitrary levels of skills at arbitrary tasks given appropriate training data.
Still, it cannot adapt to novelty or pick up new skills on the fly (which is to say that there is no fluid intelligence at play here.)
This has been exemplified by the low performance of LLMs on ARC-AGI, the only benchmark specifically designed to measure adaptability to novelty
– GPT-3 scored 0, GPT-4 scored near 0, and GPT-4o got to 5%. Scaling up these models to the limits of what's possible wasn't getting ARC-AGI numbers
anywhere near what basic brute enumeration could achieve years ago (up to 50%).
To adapt to novelty, you need two things.
First, you need knowledge – a set of reusable functions or programs to draw upon. LLMs have more than enough of that.
Second, you need the ability to recombine these functions into a brand new program when facing a new task – a program that models the task at hand.
Program synthesis. LLMs have long lacked this feature. The o series of models fixes that.
For now, we can only speculate about the exact specifics of how o3 works. However, o3's core mechanism appears to be a natural language program search and execution within the token space.
At test time, the model searches over the space of possible Chains of Thought (CoTs), describing the steps required to solve the task in a fashion perhaps not too dissimilar
to AlphaZero-style Monte-Carlo tree search. In the case of o3, the search is presumably guided by some evaluator model.
To note, Demis Hassabis hinted back in a June 2023 interview that DeepMind had been researching this idea – this line of work has been coming for a long time.

From: https://www.letta.com/blog/ai-agents-stack
> Agents are a significantly harder engineering challenge compared to basic LLM chatbots because they require state management
> (retaining the message/event history, storing long-term memories, executing multiple LLM calls in an agentic loop) and tool execution
> (safely executing an action output by an LLM and returning the result).

From: [Andriy Burkov's post](https://www.linkedin.com/posts/andriyburkov_instead-of-asking-an-llm-does-this-fragment-activity-7282199264645001216-Hmlx?utm_source=share&utm_medium=member_desktop):
> Instead of asking an LLM, "Does this fragment contain an error?" say, "This fragment contains an error. Find it."
> Then, if it finds an error, ask, "Are you sure about it?"
> If it says, "Sorry, I was wrong," then there's unlikely an error.
