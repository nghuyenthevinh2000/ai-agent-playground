import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

dotenv.config();

const llm = new ChatOpenAI({
    model: "gpt-4",
    temperature: 0.0,
})

const taggingPrompt = ChatPromptTemplate.fromTemplate(
`Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
`
);

// result schema
// {
//  sentiment,
//  aggressiveness,
//  language
// }
const classificationSchema = z.object({
    sentiment: z.string().describe("The sentiment of the text"),
    aggressiveness: z
        .number()
        .int()
        .min(1)
        .max(10)
        .describe("How aggressive the text is on a scale from 1 to 10"),
    language: z.string().describe("The language the text is written in"),
});

// Name is optional, but gives the models more clues as to what your schema represents
const llmWithStructuredOutput = llm.withStructuredOutput(classificationSchema, {
    name: "extractor",
});

// the model can infer the result for a given prompt
const prompt1 = await taggingPrompt.invoke({
    input:
        "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!",
});

let res = await llmWithStructuredOutput.invoke(prompt1);
console.log(res);

// more refined classification schema
// {
// sentiment,
// aggressiveness,
// language
// }
const classificationSchema2 = z.object({
    sentiment: z
      .enum(["happy", "neutral", "sad"])
      .describe("The sentiment of the text"),
    aggressiveness: z
      .number()
      .int()
      .min(1)
      .max(5)
      .describe("describes how aggressive the statement is, the higher the number the more aggressive"),
    language: z
      .enum(["spanish", "english", "french", "german", "italian"])
      .describe("The language the text is written in"),
});

// Holy shit! explicitly specifying the scale works for LLM
// turns out it was mesmorizing my first classification schema
const taggingPrompt2 = ChatPromptTemplate.fromTemplate(
`
    Extract the desired information from the following passage.
  
    Only extract the properties mentioned in the 'Classification' function.
    For sentiment, use only "happy", "neutral", or "sad".
    For aggressiveness, use a scale from 1 to 5, where 1 is least aggressive and 5 is most aggressive.
    For language, specify either "spanish", "english", "french", "german", or "italian".
  
    Passage:
    {input}
`
);
  
const llmWithStructuredOutput2 = llm.withStructuredOutput(classificationSchema2, { 
    name: "extractor",
});

const prompt2 = await taggingPrompt2.invoke({
    input:
      "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!",
});
let res1 = await llmWithStructuredOutput2.invoke(prompt2);
console.log(res1);

const prompt3 = await taggingPrompt2.invoke({
    input: "Estoy muy enojado con vos! Te voy a dar tu merecido!",
  });
let res2 = await llmWithStructuredOutput2.invoke(prompt3);
console.log(res2);