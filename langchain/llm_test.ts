import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages"; 
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';

// load config
dotenv.config();

// setup a ChatModels, an instance of Runnables
const model = new ChatOpenAI({ model: "gpt-4" });

// INPUT 1: direct user message to the model
const messages = [
    new SystemMessage("Translate the following from English into Italian"),
    new HumanMessage("hi!"),
];
  
await model.invoke(messages);

// INPUT 2: streaming. What are the reasons for streaming?
// Real-time processing
// Handling long outputs - chunking
// Progressing - early termination
const stream = await model.stream(messages);

const chunks = [];
for await (const chunk of stream) {
  chunks.push(chunk);
  console.log(`${chunk.content}|`);
}

// INPUT 3: prompt template - a bunch of text templates for formatting from user messages
const systemTemplate = "Translate the following from English into {language}";
const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["user", "{text}"],
]);
const promptValue = await promptTemplate.invoke({
    language: "italian",
    text: "hi!",
});

const response = await model.invoke(promptValue);
console.log(`${response.content}`);