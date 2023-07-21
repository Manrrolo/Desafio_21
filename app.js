const express = require('express');
const multer = require('multer');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const fs = require('fs');
const path = require('path');
require('dotenv').config();
const { OpenAI } = require('langchain/llms/openai');
const { FaissStore } = require('langchain/vectorstores/faiss');
const { OpenAIEmbeddings } = require('langchain/embeddings/openai');
const stream = require('stream');
const util = require('util');
const pipeline = util.promisify(stream.pipeline);

const app = express();
const upload = multer();

app.use(express.static('public'));

const server = app.listen(process.env.PORT || 3000, function () {
  console.log(`Listening on port ${server.address().port}`);
});

const io = require('socket.io')(server);

// Configure OpenAI API
const openai = new OpenAI({ temperature: 0 });

async function parsePdfToText(pdfBuffer) {
  return new Promise((resolve, reject) => {
    pdfParse(pdfBuffer)
      .then((data) => resolve(data.text))
      .catch((err) => reject(err));
  });
}

async function getVoiceAudio(text, voiceId = 'ErXwobaYiN019PkySvjV') {
  const tempy = await import('tempy');
  const CHUNK_SIZE = 1024;
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`;

  const headers = {
    Accept: 'audio/mpeg',
    'Content-Type': 'application/json',
    'xi-api-key': '62e2fd73469f1eac0523ff39cd1489d8',
  };

  const data = {
    text: text,
    model_id: 'eleven_multilingual_v1',
    voice_settings: {
      stability: 0.4,
      similarity_boost: 1.0,
    },
  };

  const response = await axios.post(url, data, {
    headers: headers,
    responseType: 'stream',
  });

  // Save audio data to a temporary file
  const tempFileName = tempy.file({ extension: 'mp3' });
  const writer = fs.createWriteStream(tempFileName);

  await pipeline(response.data, writer);

  return tempFileName;
}

app.post('/upload', upload.single('pdf'), async (req, res) => {
  // You can handle pdf upload and parsing here
  let dataBuffer = req.file.buffer;

  pdfParse(dataBuffer).then(async function (data) {
    let text = data.text;

    // Chunk size and overlap in terms of characters
    let chunkSize = 1000;
    let chunkOverlap = 200;

    let chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
      let end = Math.min(i + chunkSize, text.length);
      let chunk = text.slice(i, end);
      chunks.push(chunk);
      i -= chunkOverlap; // Overlap chunks
    }

    let storeName = req.file.originalname.split('.')[0];
    let pathToFile = path.join(__dirname, storeName + '.json');
    let vectorStore;

    if (fs.existsSync(pathToFile)) {
      // If the file exists, load the VectorStore from the file
      let data = fs.readFileSync(pathToFile, 'utf8');
      vectorStore = await FaissStore.fromJSON(JSON.parse(data));
    } else {
      // If the file does not exist, create a new VectorStore from the chunks
      let docs = chunks.map((text, id) => {
        return { id: id, text: text };
      });
      vectorStore = await FaissStore.fromDocuments(
        docs,
        new OpenAIEmbeddings(),
      );

      // Then, save the VectorStore to a JSON file for future use
      let jsonData = vectorStore.toJSON();
      let stringData = JSON.stringify(jsonData);
      fs.writeFileSync(pathToFile, stringData);
    }

    res.send('PDF uploaded and processed successfully.');
  });
});

io.on('connection', (socket) => {
  socket.on('ask', async (question) => {
    // Get the storeName from the client, assuming it's sent along with the question
    let storeName = question.storeName;
    question = question.text;

    // Load the VectorStore from a file
    let pathToFile = path.join(__dirname, storeName + '.json');
    let data = fs.readFileSync(pathToFile, 'utf8');
    let vectorStore = await FaissStore.fromJSON(JSON.parse(data));

    // Search for the most similar documents
    let docs = await vectorStore.similaritySearch(question, 3);

    // Extract the texts from the docs
    let inputDocuments = docs.map((doc) => doc.text);

    // Generate the prompt for the OpenAI API
    let prompt =
      inputDocuments.join('\n') + '\n' + question + ' Responde en español';

    // Call the OpenAI API
    const gptResponse = await openai.complete({
      engine: 'gpt-3.5-turbo',
      prompt: prompt,
      max_tokens: 150,
      temperature: 0,
    });

    // Send the OpenAI response back to the client
    let response = gptResponse.data.choices[0].text.trim();
    socket.emit('response', response);

    // If useAudio is enabled, generate and send audio
    if (socket.handshake.query.useAudio) {
      let audioFile = await getVoiceAudio(response);
      socket.emit('audio', audioFile);
    }
  });
});
