import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import z from "zod";

dotenv.config();

// output schema for structured data
const personSchema = z.object({
    name: z.optional(z.string()).describe("The name of the person"),
    hair_color: z
      .optional(z.string())
      .describe("The color of the person's hair if known"),
    height_in_meters: z
        .number()
        .nullish()
        .describe("Height measured in meters"),
});

const dataSchema = z.object({
    people: z.array(personSchema).describe("Extracted data about people"),
});

// Define a custom prompt to provide instructions and any additional context.
// 1) You can add examples into the prompt template to improve extraction quality
// 2) Introduce additional parameters to take context into account (e.g., include metadata
//    about the document from which the text was extracted.)
// interesting how the added prompt can change so much the reasoning and performance
const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert extraction algorithm.
  Only extract relevant information from the text.
  If you see height measured in any other unit, convert it to meters.
  If you do not know the value of an attribute asked to extract,
  return null for the attribute's value.`,
    ],
    // Please see the how-to about improving performance with
    // reference examples.
    // ["placeholder", "{examples}"],
    ["human", "{text}"],
]);

// setup a ChatModels, an instance of Runnables
const llm = new ChatOpenAI({ 
    model: "gpt-4",
    temperature: 0.7,
});
const structured_llm = llm.withStructuredOutput(personSchema);

const prompt = await promptTemplate.invoke({
    text: "Alan Smith is 6 feet tall and has blond hair.",
});
let res = await structured_llm.invoke(prompt);
console.log(res);

const structured_llm2 = llm.withStructuredOutput(personSchema, {
    name: "person",
});
  
const prompt2 = await promptTemplate.invoke({
    text: "Alan Smith is 6 feet tall and has blond hair.",
});
let res1 = await structured_llm2.invoke(prompt2);
console.log(res1);

const structured_llm3 = llm.withStructuredOutput(dataSchema);
const prompt3 = await promptTemplate.invoke({
  text: "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me.",
});
let res2 = await structured_llm3.invoke(prompt3);
console.log(res2);