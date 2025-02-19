#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Сигмоидальная функция активации
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная сигмоидальной функции
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
private:
    vector<vector<double>> weights_input_hidden;
    vector<vector<double>> weights_hidden_output;
    vector<double> hidden_layer;
    vector<double> output_layer;

    double learning_rate;

public:
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, double lr) {
        learning_rate = lr;

        // Инициализация весов случайными значениями
        weights_input_hidden.resize(input_nodes, vector<double>(hidden_nodes));
        weights_hidden_output.resize(hidden_nodes, vector<double>(output_nodes));

        for (int i = 0; i < input_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                weights_input_hidden[i][j] = ((double) rand() / (RAND_MAX)) * 2 - 1;
            }
        }

        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                weights_hidden_output[i][j] = ((double) rand() / (RAND_MAX)) * 2 - 1;
            }
        }

        hidden_layer.resize(hidden_nodes);
        output_layer.resize(output_nodes);
    }

    vector<double> predict(const vector<double>& inputs) {
        // Прямое распространение
        for (size_t i = 0; i < hidden_layer.size(); ++i) {
            hidden_layer[i] = 0.0;
            for (size_t j = 0; j < inputs.size(); ++j) {
                hidden_layer[i] += inputs[j] * weights_input_hidden[j][i];
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);
        }

        for (size_t i = 0; i < output_layer.size(); ++i) {
            output_layer[i] = 0.0;
            for (size_t j = 0; j < hidden_layer.size(); ++j) {
                output_layer[i] += hidden_layer[j] * weights_hidden_output[j][i];
            }
            output_layer[i] = sigmoid(output_layer[i]);
        }

        return output_layer;
    }

    void train(const vector<double>& inputs, const vector<double>& targets) {
        // Прямое распространение
        vector<double> outputs = predict(inputs);

        // Обратное распространение ошибки
        vector<double> output_errors(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            output_errors[i] = targets[i] - outputs[i];
        }

        vector<double> hidden_errors(hidden_layer.size(), 0.0);
        for (size_t i = 0; i < hidden_layer.size(); ++i) {
            for (size_t j = 0; j < output_errors.size(); ++j) {
                hidden_errors[i] += output_errors[j] * weights_hidden_output[i][j];
            }
        }

        // Обновление весов между скрытым и выходным слоем
        for (size_t i = 0; i < hidden_layer.size(); ++i) {
            for (size_t j = 0; j < output_errors.size(); ++j) {
                weights_hidden_output[i][j] += learning_rate * output_errors[j] * sigmoid_derivative(outputs[j]) * hidden_layer[i];
            }
        }

        // Обновление весов между входным и скрытым слоем
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < hidden_errors.size(); ++j) {
                weights_input_hidden[i][j] += learning_rate * hidden_errors[j] * sigmoid_derivative(hidden_layer[j]) * inputs[i];
            }
        }
    }
};

int main() {
    srand(time(0));

    // Создаем нейронную сеть с 2 входными нейронами, 3 нейронами в скрытом слое и 1 выходным нейроном
    NeuralNetwork nn(2, 3, 1, 0.1);

    // Обучаем сеть на примере XOR
    vector<vector<double>> training_inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> training_targets = {{0}, {1}, {1}, {0}};

    for (int i = 0; i < 10000; ++i) {
        for (size_t j = 0; j < training_inputs.size(); ++j) {
            nn.train(training_inputs[j], training_targets[j]);
        }
    }

    cout << "Нейронная сеть обучена на данных XOR." << endl;
    cout << "Введите два числа (0 или 1) для предсказания результата:" << endl;

    while (true) {
        double input1, input2;
        cout << "Введите первое число: ";
        cin >> input1;
        cout << "Введите второе число: ";
        cin >> input2;

        if (input1 != 0 && input1 != 1) {
            cout << "Ошибка: введите 0 или 1." << endl;
            continue;
        }
        if (input2 != 0 && input2 != 1) {
            cout << "Ошибка: введите 0 или 1." << endl;
            continue;
        }

        vector<double> inputs = {input1, input2};
        vector<double> output = nn.predict(inputs);

        cout << "Результат предсказания: " << output[0] << endl;
        cout << "Сеть считает, что результат ближе к " << (output[0] > 0.5 ? "1" : "0") << endl;

        char choice;
        cout << "Хотите продолжить? (y/n): ";
        cin >> choice;
        if (choice != 'y' && choice != 'Y') {
            break;
        }
    }

    cout << "Программа завершена." << endl;
    return 0;
}
