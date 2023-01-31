#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::cout;
using std::vector;
using std::sort;
using std::exp;
using std::min;
using std::max;
using std::ofstream;

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, sizeof(result), nanosecond());
		result = Hash((uint8_t*)&result, sizeof(result), microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	uint32_t Ruint32()	// XORSHIFT128+
	{
		uint64_t a = state[0];
		uint64_t b = state[1];
		state[0] = b;
		a ^= a << 23;
		state[1] = a ^ b ^ (a >> 18) ^ (b >> 5);
		return uint32_t((state[1] + b) >> 16);
	}

	float Rfloat(float min = 0, float max = 1) { return min + (max - min) * Ruint32() * 2.3283064371e-10; }

	static uint32_t Hash(const uint8_t* key, size_t len, uint32_t seed = 0)	// MurmurHash3
	{
		uint32_t h = seed;
		uint32_t k;
		for (size_t i = len >> 2; i; i--) {
			memcpy(&k, key, sizeof(uint32_t));
			key += sizeof(uint32_t);
			h ^= murmur_32_scramble(k);
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}
		k = 0;
		for (size_t i = len & 3; i; i--) {
			k <<= 8;
			k |= key[i - 1];
		}
		h ^= murmur_32_scramble(k);
		h ^= len;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

private:
	uint64_t state[2];

	static uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	static uint32_t nanosecond() { return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
	static uint32_t microsecond() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
};

namespace GlobalVars
{
	Random random(Random::MakeSeed(0));
	constexpr uint32_t INPUT = 2;
	constexpr uint32_t HIDDEN = 4;
	constexpr uint32_t OUTPUT = 2;
	constexpr float ONE = 1.0f;
	constexpr float ZERO = 0.0f;
	constexpr float LEARNING_RATE = 0.01f;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min, float max)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GlobalVars::random.Rfloat(min, max);
}

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuCLU(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	for (size_t counter = size; counter--;)
		outputMatrix[counter] = min(1.0f, max(-1.0f, inputMatrix[counter]));
}

const static void cpuCLUGradient(float* inputMatrix, float* gradientMatrix, float* outputMatrix, uint32_t size) {
	float input;
	float gradient;
	bool greaterZero;
	for (size_t counter = size; counter--;)
	{
		input = inputMatrix[counter];
		gradient = gradientMatrix[counter];
		greaterZero = gradient > 0;
		gradient = (greaterZero << 1) - 1;
		outputMatrix[counter] = (((input >= 1) ^ greaterZero) || ((input > -1) ^ greaterZero)) * gradient;
	}
}

void cpuSoftmax(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		outputMatrix[counter] = exp(inputMatrix[counter]);
		sum += outputMatrix[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		outputMatrix[counter] *= sum;
}

void cpuSoftmaxGradient(float* outputMatrix, bool isSurvivor, uint32_t action, float* resultMatrix, uint32_t size)
{
	int agentGradient = (isSurvivor << 1) - 1;
	/*float sampledProbability = outputMatrix[action];
	for (uint32_t counter = size; counter--;)
			resultMatrix[counter] = agentGradient * outputMatrix[counter] * ((counter == action) - sampledProbability);
	*/
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = (((counter == action) << 1) - 1) * agentGradient;
}

void cpuLinearlize(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		outputMatrix[counter] = (inputMatrix[counter] > 0) ? 1 : -1;
}

int main()
{
	const bool debug = false;
	float scores[100] = { 0 };
	uint32_t idx = 0;
	float avgScore = 0;
	
	float inputMatrix[GlobalVars::INPUT];
	float hiddenMatrix[GlobalVars::HIDDEN];
	float cluHiddenMatrix[GlobalVars::HIDDEN];
	float outputMatrix[GlobalVars::OUTPUT];
	//float cluOutputMatrix[GlobalVars::OUTPUT];
	float softmaxMatrix[GlobalVars::OUTPUT];

	float inputHiddenWeights[GlobalVars::INPUT * GlobalVars::HIDDEN];
	float hiddenOutputWeights[GlobalVars::HIDDEN * GlobalVars::OUTPUT];

	float hiddenBias[GlobalVars::HIDDEN];
	float outputBias[GlobalVars::OUTPUT];
	
	float inputHiddenWeightsGradient[GlobalVars::INPUT * GlobalVars::HIDDEN];
	float hiddenOutputWeightsGradient[GlobalVars::HIDDEN * GlobalVars::OUTPUT];

	float hiddenBiasGradient[GlobalVars::HIDDEN];
	float outputBiasGradient[GlobalVars::OUTPUT];

	cpuGenerateUniform(inputHiddenWeights, GlobalVars::INPUT * GlobalVars::HIDDEN, -1.0f, 1.0f);
	cpuGenerateUniform(hiddenOutputWeights, GlobalVars::HIDDEN * GlobalVars::OUTPUT, -1.0f, 1.0f);

	cpuGenerateUniform(hiddenBias, GlobalVars::HIDDEN, -1.0f, 1.0f);
	cpuGenerateUniform(outputBias, GlobalVars::OUTPUT, -1.0f, 1.0f);

	uint32_t iteration = 4000;
	while (iteration--)
	{
		uint32_t input1 = GlobalVars::random.Ruint32() & 1;
		uint32_t input2 = GlobalVars::random.Ruint32() & 1;

		if (GlobalVars::random.Ruint32() & 1)
		{
			input1 = 1;
		}

		if (iteration == 0)
		{
			input1 = 1;
			input2 = 1;
		}
		uint32_t expected = input1 ^ input2;
		
		inputMatrix[0] = input1;
		inputMatrix[1] = input2;

		if (debug)
		{
			cout << "Input Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::INPUT; counter++)
				cout << inputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';

			cout << "Input Hidden Weights:\n";
			for (uint32_t counter = 0; counter < GlobalVars::INPUT; counter++)
			{
				for (uint32_t counter2 = 0; counter2 < GlobalVars::HIDDEN; counter2++)
					cout << inputHiddenWeights[counter * GlobalVars::HIDDEN + counter2] << ' ';
				cout << '\n';
			}
			cout << '\n';
		}

		cpuSgemmStridedBatched(
			false, false,
			GlobalVars::HIDDEN, 1, GlobalVars::INPUT,
			&GlobalVars::ONE,
			inputHiddenWeights, GlobalVars::HIDDEN, GlobalVars::ZERO,
			inputMatrix, GlobalVars::INPUT, GlobalVars::ZERO,
			&GlobalVars::ZERO,
			hiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
			1);

		if (debug)
		{
			cout << "Hidden Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
				cout << hiddenMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';

			cout << "Hidden Bias:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
				cout << hiddenBias[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		for (uint32_t counter = GlobalVars::HIDDEN; counter--;)
			hiddenMatrix[counter] += hiddenBias[counter];

		if (debug)
		{
			cout << "Hidden Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
				cout << hiddenMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		cpuCLU(hiddenMatrix, cluHiddenMatrix, GlobalVars::HIDDEN);

		if (debug)
		{
			cout << "CLU Hidden Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
				cout << cluHiddenMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';

			cout << "Hidden Output Weights:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
			{
				for (uint32_t counter2 = 0; counter2 < GlobalVars::OUTPUT; counter2++)
					cout << hiddenOutputWeights[counter * GlobalVars::OUTPUT + counter2] << ' ';
				cout << '\n';
			}
			cout << '\n';
		}
		
		cpuSgemmStridedBatched(
			false, false,
			GlobalVars::OUTPUT, 1, GlobalVars::HIDDEN,
			&GlobalVars::ONE,
			hiddenOutputWeights, GlobalVars::OUTPUT, GlobalVars::ZERO,
			cluHiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
			&GlobalVars::ZERO,
			outputMatrix, GlobalVars::OUTPUT, GlobalVars::ZERO,
			1);

		if (debug)
		{
			cout << "Output Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << outputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';

			cout << "Output Bias:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << outputBias[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		for (uint32_t counter = GlobalVars::OUTPUT; counter--;)
			outputMatrix[counter] += outputBias[counter];

		if (debug)
		{
			cout << "Output Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << outputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		//cpuCLU(outputMatrix, cluOutputMatrix, GlobalVars::OUTPUT);

		/*if (debug)
		{
			cout << "CLU Output Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << cluOutputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}*/
		
		//cpuSoftmax(cluOutputMatrix, softmaxMatrix, GlobalVars::OUTPUT);
		cpuSoftmax(outputMatrix, softmaxMatrix, GlobalVars::OUTPUT);

		if (debug)
		{
			cout << "Softmax Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << softmaxMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		float number = GlobalVars::random.Rfloat(0.0f, 1.0f);
		uint32_t action = 0;
		while (true)
		{
			number -= softmaxMatrix[action];
			if (number < 0) break;
			action++;
			action -= (action == GlobalVars::OUTPUT) * GlobalVars::OUTPUT;
		}

		if (debug)
		{
			cout << "Input: " << input1 << " " << input2 << '\n';
			cout << "Expected: " << expected << '\n';
			cout << "Output: " << action << '\n';
			cout << "Probability: ";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << softmaxMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		avgScore -= scores[idx];
		avgScore += (action == expected);
		scores[idx++] = (action == expected);
		idx -= (idx == 100) * 100;
		
		cout << "Score: " << avgScore / 100 << '\n';
		
		//cpuSoftmaxGradient(softmaxMatrix, action == expected, action, cluOutputMatrix, GlobalVars::OUTPUT);
		/*if (debug)
		{
			cout << "Softmax Gradient:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << cluOutputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}*/
		
		cpuSoftmaxGradient(softmaxMatrix, action == expected, action, outputMatrix, GlobalVars::OUTPUT);
		
		if (debug)
		{
			cout << "Softmax Gradient:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << outputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		//cpuCLUGradient(outputMatrix, cluOutputMatrix, outputMatrix, GlobalVars::OUTPUT);

		/*if (debug)
		{
			cout << "Output Gradient:\n";
			for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
				cout << outputMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}*/
		
		cpuSgemmStridedBatched(
			false, true,
			GlobalVars::OUTPUT, GlobalVars::HIDDEN, 1,
			&GlobalVars::ONE,
			outputMatrix, GlobalVars::OUTPUT, GlobalVars::ZERO,
			hiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
			&GlobalVars::ZERO,
			hiddenOutputWeightsGradient, GlobalVars::OUTPUT, GlobalVars::ZERO,
			1);

		cpuLinearlize(hiddenOutputWeightsGradient, hiddenOutputWeightsGradient, GlobalVars::HIDDEN * GlobalVars::OUTPUT);


		if (debug)
		{
			cout << "Hidden Output Weights Gradient:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
			{
				for (uint32_t counter2 = 0; counter2 < GlobalVars::OUTPUT; counter2++)
					cout << hiddenOutputWeightsGradient[counter * GlobalVars::OUTPUT + counter2] << ' ';
				cout << '\n';
			}
			cout << '\n';
		}
		
		cpuSgemmStridedBatched(
			true, false,
			GlobalVars::HIDDEN, 1, GlobalVars::OUTPUT,
			&GlobalVars::ONE,
			hiddenOutputWeights, GlobalVars::OUTPUT, GlobalVars::ZERO,
			outputMatrix, GlobalVars::OUTPUT, GlobalVars::ZERO,
			&GlobalVars::ZERO,
			cluHiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
			1);

		if (debug)
		{
			cout << "CLU Hidden Matrix:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
				cout << cluHiddenMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		cpuCLUGradient(hiddenMatrix, cluHiddenMatrix, hiddenMatrix, GlobalVars::HIDDEN);

		if (debug)
		{
			cout << "Hidden Gradient:\n";
			for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
				cout << hiddenMatrix[counter] << ' ';
			cout << '\n';
			cout << '\n';
		}
		
		cpuSgemmStridedBatched(
			false, true,
			GlobalVars::HIDDEN, GlobalVars::INPUT, 1,
			&GlobalVars::ONE,
			hiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
			inputMatrix, GlobalVars::INPUT, GlobalVars::ZERO,
			&GlobalVars::ZERO,
			inputHiddenWeightsGradient, GlobalVars::HIDDEN, GlobalVars::ZERO,
			1);

		cpuLinearlize(inputHiddenWeightsGradient, inputHiddenWeightsGradient, GlobalVars::INPUT * GlobalVars::HIDDEN);
		
		if (debug)
		{
			cout << "Input Hidden Weights Gradient:\n";
			for (uint32_t counter = 0; counter < GlobalVars::INPUT; counter++)
			{
				for (uint32_t counter2 = 0; counter2 < GlobalVars::HIDDEN; counter2++)
					cout << inputHiddenWeightsGradient[counter * GlobalVars::HIDDEN + counter2] << ' ';
				cout << '\n';
			}
			cout << '\n';
		}
		
		/*cpuSgemmStridedBatched(
			true, false,
			GlobalVars::INPUT, 1, GlobalVars::HIDDEN,
			&GlobalVars::ONE,
			inputHiddenWeights, GlobalVars::HIDDEN, GlobalVars::ZERO,
			hiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
			&GlobalVars::ZERO,
			inputMatrix, GlobalVars::INPUT, GlobalVars::ZERO,
			1);*/

		for (uint32_t counter = 0; counter < GlobalVars::INPUT * GlobalVars::HIDDEN; counter++)
		{
			inputHiddenWeights[counter] += inputHiddenWeightsGradient[counter] * GlobalVars::LEARNING_RATE;
		}

		for (uint32_t counter = 0; counter < GlobalVars::HIDDEN * GlobalVars::OUTPUT; counter++)
		{
			hiddenOutputWeights[counter] += hiddenOutputWeightsGradient[counter] * GlobalVars::LEARNING_RATE;
		}

		for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
		{
			hiddenBias[counter] += hiddenMatrix[counter] * GlobalVars::LEARNING_RATE;
		}
		
		for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
		{
			outputBias[counter] += outputMatrix[counter] * GlobalVars::LEARNING_RATE;
		}
	}
	cout << '\n';
	
	cout << "Input Hidden Weights:\n";
	for (uint32_t counter = 0; counter < GlobalVars::INPUT; counter++)
	{
		for (uint32_t counter2 = 0; counter2 < GlobalVars::HIDDEN; counter2++)
			cout << inputHiddenWeights[counter * GlobalVars::HIDDEN + counter2] << ' ';
		cout << '\n';
	}
	cout << '\n';

	cout << "Hidden Bias:\n";
	for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
		cout << hiddenBias[counter] << ' ';
	cout << '\n';
	cout << '\n';

	cout << "Hidden Output Weights:\n";
	for (uint32_t counter = 0; counter < GlobalVars::HIDDEN; counter++)
	{
		for (uint32_t counter2 = 0; counter2 < GlobalVars::OUTPUT; counter2++)
			cout << hiddenOutputWeights[counter * GlobalVars::OUTPUT + counter2] << ' ';
		cout << '\n';
	}
	cout << '\n';

	cout << "Output Bias:\n";
	for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
		cout << outputBias[counter] << ' ';
	cout << '\n';
	cout << '\n';

	// print the input and softmax output
	cout << "Input:\n";
	for (uint32_t counter = 0; counter < GlobalVars::INPUT; counter++)
		cout << inputMatrix[counter] << ' ';
	cout << '\n';
	cout << '\n';

	cout << "Softmax Output:\n";
	for (uint32_t counter = 0; counter < GlobalVars::OUTPUT; counter++)
		cout << softmaxMatrix[counter] << ' ';
	cout << '\n';
	cout << '\n';
	
	return 0;
}