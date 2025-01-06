import dotenv from "dotenv";
import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

dotenv.config();

// DATA FORMAT 1: Document abstraction represents a unit of text and associated metadata
// 1_0: a sample document with metadata
const documents = [
    new Document({
      pageContent:
        "Dogs are great companions, known for their loyalty and friendliness.",
      metadata: { source: "mammal-pets-doc" },
    }),
    new Document({
      pageContent: "Cats are independent pets that often enjoy their own space.",
      metadata: { source: "mammal-pets-doc" },
    }),
];

// 1_1: Loading documents
const loader = new PDFLoader("data/BTC liquidity layer.pdf");

const docs = await loader.load();
console.log(docs.length);

docs[0].pageContent.slice(0, 200);
docs[0].metadata;

// 1_2: Splitting documents into smaller chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

allSplits.length;

// DATA FORMAT 2: Embeddings from providers
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

console.assert(vector1.length === vector2.length);
console.log(`Generated vectors of length ${vector1.length}\n`);
console.log(vector1.slice(0, 10));

// DATA STORAGE 1: Vector stores contain text and Document objects, stored in Embedding format
const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(allSplits);

// DS_1_1: vector stores have different ways of CRUD compared to normal databases
const results1 = await vectorStore.similaritySearch(
  "Bitcoin liquidity layer"
);

console.log("DS_1_1: ", results1[0]);

// DS_1_2: similarity search with score shows how relevant the two texts are
const results2 = await vectorStore.similaritySearchWithScore(
  "FROST technology depends on Schnorr signatures "
);

console.log("DS_1_2: ", results2[0]);

// DS_1_3: similarity search with embedding directly instead of string
const embedding = await embeddings.embedQuery(
  "Bitcoin Maxi community is very skeptical of new Taproot technology"
);

const results3 = await vectorStore.similaritySearchVectorWithScore(
  embedding,
  1
);

console.log("DS_1_3: ", results3[0]);

// DATA STORAGE RETRIEVER: complex retriever for complex retrieval logic
// Should also learn various retrieval strategies.
const retriever = vectorStore.asRetriever({
  searchType: "mmr",
  searchKwargs: {
    fetchK: 1,
  },
});

let res = await retriever.batch([
  "FROST is a multi signatures",
  "DKG is a component of FROST",
]);

console.log("DS_RETRIEVER: ", res);