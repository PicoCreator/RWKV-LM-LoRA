#!/usr/bin/env node

// Dependencies
const fs = require('fs');
const path = require('path');
const randomWords = require('random-words');
const process = require('process');

// Check for correct number of arguments
if (process.argv.length !== 5) {
	console.log('Usage: node script.js <outputFilePath> <maxWords> <numSamples>');
	process.exit(1);
}
const [outputFilePath, maxWords, numSamples] = process.argv.slice(2);

// The main function
async function generateJsonl(outputFilePath, maxWords, numSamples) {
	const filePath = path.resolve(outputFilePath);
	const writeStream = fs.createWriteStream(filePath);

	const promptTemplates = [
		"Simon:\n```\n{{document}}\n```\n\nSays:",
		"Memorize the following document:\n```\n{{document}}```\n\nType it out below:",
		"Memorise and reply back with the following document:\n```\n{{document}}```\n\nReply:",
	];
	const completionTemplates = [
		"\n```\n{{document}}\n```\n",
		"\n```\n{{document}}\n```\n",
		"\n```\n{{document}}\n```\n",
	];

	const getRandomInt = (min, max) => {
		return Math.floor(Math.random() * (max - min + 1)) + min;
	};

	const getRandomPromptCompletionPair = () => {
		let documentArr = [];
		let wordCount = 0;

		// Generate random paragraphs, each with 100 words max
		// And merge it into a single document
		for(let i = 0; i < maxWords; i += 100) {
			let paragraphMax = Math.min( maxWords - wordCount, 100 );

			// This is intentionally biased towards the paragraphMax
			let paragraph = randomWords({ 
				min: getRandomInt(paragraphMax/2, paragraphMax), 
				max: paragraphMax 
			});

			wordCount += paragraphMax;
			documentArr.push(paragraph.join(' '));
			documentArr.push("\n\n")
		}
		const document = documentArr.join('');

		// Pick a random template and fill it in with the document
		const templateIndex = getRandomInt(0, promptTemplates.length - 1);
		const prompt = promptTemplates[templateIndex].replace('{{document}}', document);
		const completion = completionTemplates[templateIndex].replace('{{document}}', document);

		// The prompt completion pair
		return { 
			prompt: prompt, 
			completion: completion
		};
	};

	try {
		for (let i = 0; i < numSamples; i++) {
			const pair = getRandomPromptCompletionPair();
			writeStream.write(JSON.stringify(pair) + '\n');
		}
	} catch (err) {
		console.error('Error writing to file:', err);
		process.exit(2);
	} finally {
		writeStream.end();
	}

	console.log(`Generated JSONL file with ${numSamples} samples at ${filePath}`);
}

// Run the main function
generateJsonl(outputFilePath, parseInt(maxWords), parseInt(numSamples));