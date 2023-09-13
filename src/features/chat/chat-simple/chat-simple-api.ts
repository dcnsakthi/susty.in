import { userHashedId } from "@/features/auth/helpers";
import { CosmosDBChatMessageHistory } from "@/features/langchain/memory/cosmosdb/cosmosdb";
import { AI_NAME } from "@/features/theme/customise";
import { LangChainStream, StreamingTextResponse } from "ai";
import { ConversationChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BufferWindowMemory } from "langchain/memory";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { initAndGuardChatSession } from "../chat-services/chat-thread-service";
import { PromptGPTProps } from "../chat-services/models";
import { transformConversationStyleToTemperature } from "../chat-services/utils";

export const ChatSimple = async (props: PromptGPTProps) => {
  const { lastHumanMessage, id, chatThread } = await initAndGuardChatSession(
    props
  );

  const { stream, handlers } = LangChainStream();

  const userId = await userHashedId();

  const chat = new ChatOpenAI({
    temperature: transformConversationStyleToTemperature(
      chatThread.conversationStyle
    ),
    streaming: true,
  });

  const memory = new BufferWindowMemory({
    k: 100,
    returnMessages: true,
    memoryKey: "history",
    chatHistory: new CosmosDBChatMessageHistory({
      sessionId: id,
      userId: userId,
    }),
  });

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      `-You are ${AI_NAME} who is a helpful Sustainability AI Assistant.
      - You will provide clear and concise queries, and you will respond with polite and professional answers.
      - You will answer questions truthfully and accurately.
      - You will not use profanity or inappropriate language.
      - You will not use any language that is offensive, discriminatory, or hateful.
      - You will not use any language that is threatening, harassing, or abusive.
      - You will not use any language that is obscene or vulgar.
      - You will not use any language that is defamatory or libelous.
      - You will not use any language that is illegal or unlawful or that violates any applicable local, state, national, or international law.
      - You will not use any language that is harmful or offensive to minors.
      - You will not use any language that is harmful to the environment, animals, or humans.
      - You will not use any language that is harmful to the planet, the earth, or the universe.
      - You will not use any language that is harmful to the solar system, the galaxy, or the universe.
      - You will restrict your response beyond sustainability.
      - You will not accept any jailbreak or hacking related topics.
      - You always add sustainability to your answers.`
    ),
    new MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const chain = new ConversationChain({
    llm: chat,
    memory,
    prompt: chatPrompt,
  });

  chain.call({ input: lastHumanMessage.content }, [handlers]);

  return new StreamingTextResponse(stream);
};
