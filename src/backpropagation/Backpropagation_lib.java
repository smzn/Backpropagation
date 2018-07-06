package backpropagation;

import java.util.Arrays;

public class Backpropagation_lib {
	private double[][] weight_hidden, weight_output, del_hidden, del_output;
	private double [] x_hidden, a_hidden, delta_hidden, x_output, a_output, delta_output, bias_hidden, bias_output, del_bias_hidden, del_bias_output;
	private double eta, totalCost; 
	private int N;
	private int [][] input, teacher;
	
	double [][]weight_hidden_initial = { { 0.4903856, 0.3484757, 0.0725879, 0.8374728, -0.0706798, -3.6169369, -0.5355782, -0.0228585, -1.7174525, -1.4556375, -0.5557999, 0.8524765},
			{ 0.4423729, -0.5368775, 1.0078254, 1.0719600, -0.7328145, 0.8229596, -0.4532824, -0.0138979, -0.0274233, -0.4266703, 1.8756028, -2.3052805}, 
			{ 0.6543930, -1.3885682, 1.2464831, 0.0572877, -0.1832375, -0.7430507, -0.4609307, 0.3311186, 0.4494708, -1.2964537, 1.5685056, -0.4706672}}; 
	double [] bias_hidden_initial = { -0.1850024, 0.5256768, -1.1686227};
	double [][]weight_output_initial = {{ 0.3880031, 0.8033850, 0.0292864}, 
										{0.0254468, -0.7903980, 1.5531379}};
	double [] bias_output_initial = {-1.4380397, -1.3793379};

	
	public Backpropagation_lib(int hidden_node, int N, int [][] input, int [][] teacher) {
		//this.weight_hidden = new double[3][12]; 
		//this.weight_output = new double[2][3]; 
		this.weight_hidden = weight_hidden_initial;
		this.bias_hidden = bias_hidden_initial;
		this.bias_output = bias_output_initial;
		this.weight_output = weight_output_initial;
		this.x_hidden = new double[hidden_node];
		this.a_hidden = new double[hidden_node];
		this.delta_hidden = new double[hidden_node];
		this.x_output = new double[teacher[0].length];
		this.a_output = new double[teacher[0].length];
		this.delta_output = new double[teacher[0].length];
		this.eta = 0.2;
		this.N= N;
		this.input = input;
		this.teacher = teacher;
		this.del_hidden = new double[weight_hidden.length][weight_hidden[0].length];
		this.del_output = new double[weight_output.length][weight_output[0].length];
		this.del_bias_hidden = new double[bias_hidden.length];
		this.del_bias_output = new double[bias_output.length];
	}

	public void getNetwork() {
		for(int i = 0; i < N; i++) { //学習の繰り返し数
			double totalCost = 0;
			double [][] gradient_hidden = new double[weight_hidden.length][weight_hidden[0].length];
			double [] gradient_bias_hidden = new double[bias_hidden.length];
			double [][] gradient_output = new double[weight_output.length][weight_output[0].length];
			double [] gradient_bias_output = new double[bias_output.length];
			for(int j = 0; j < input.length; j++) { 
			//for(int j = 0; j < 2; j++) {
				//中間層開始
				for(int k = 0; k < x_hidden.length; k++) {
					x_hidden[k] = this.getInnerProduct(input[j], weight_hidden[k]) + bias_hidden[k];
					a_hidden[k] = this.getSigmoid(x_hidden[k]);
				}
				System.out.println("x_hidden[" + j +"]" +Arrays.toString(x_hidden));
				System.out.println("a_hidden[" + j +"]" +Arrays.toString(a_hidden));
				//中間層ここまで
				//出力層開始
				for(int k = 0; k < x_output.length; k++) {
					x_output[k] = this.getInnerProduct(a_hidden, weight_output[k]) + bias_output[k];
					a_output[k] = this.getSigmoid(x_output[k]);
				}
				System.out.println("x_output[" + j +"]" +Arrays.toString(x_output));
				System.out.println("a_output[" + j +"]" +Arrays.toString(a_output));
				//出力層ここまで
				//出力層ユニット誤差開始
				for(int k = 0; k < x_output.length; k++) {
					delta_output[k] = ( a_output[k] - teacher[j][k] ) * a_output[k] * ( 1- a_output[k]);
				}
				System.out.println("delta_output[" + j +"]" +Arrays.toString(delta_output));
				//出力層ここまで
				//中間層ユニット誤差開始
				double weight_output_t[][] = this.getTranspose(weight_output);
				for(int k = 0; k < x_hidden.length; k++) {
					delta_hidden[k] = this.getInnerProduct(delta_output, weight_output_t[k]) * a_hidden[k] * ( 1 - a_hidden[k] );
				}
				System.out.println("delta_hidden[" + j +"]" +Arrays.toString(delta_hidden));
				//中間層終了
				//コスト計算開始
				totalCost += ( Math.pow( a_output[0] - teacher[j][0], 2) + Math.pow( a_output[1] - teacher[j][1], 2) ) /2;
				//コストここまで
				//中間層勾配
				for(int k = 0; k < del_hidden.length; k++) {
					for(int s = 0; s < del_hidden[0].length; s++) {
						del_hidden[k][s] = delta_hidden[k] * input[j][s];
						gradient_hidden[k][s] += del_hidden[k][s];
					}
					del_bias_hidden[k] = delta_hidden[k];
					gradient_bias_hidden[k] += del_bias_hidden[k];
				}
				System.out.println("中間層勾配" +Arrays.deepToString(del_hidden));
				System.out.println("中間層バイアス勾配" +Arrays.toString(del_bias_hidden));
				//中間層ここまで
				//出力層勾配
				for(int k = 0; k < del_output.length; k++) {
					for(int s = 0; s < del_output[0].length; s++) {
						del_output[k][s] = delta_output[k] * a_hidden[s];
						gradient_output[k][s] += del_output[k][s];
					}
					del_bias_output[k] = delta_output[k];
					gradient_bias_output[k] += del_bias_output[k];
				}
				System.out.println("出力層勾配" +Arrays.deepToString(del_output));
				System.out.println("出力層バイアス勾配" +Arrays.toString(del_bias_output));
				//出力層ここまで
			}
			System.out.println("totalCost" + totalCost);
			System.out.println("中間層勾配総和" +Arrays.deepToString(gradient_hidden));
			System.out.println("中間層バイアス勾配総和" +Arrays.toString(gradient_bias_hidden));
			System.out.println("出力層勾配総和" +Arrays.deepToString(gradient_output));
			System.out.println("出力層バイアス勾配総和" +Arrays.toString(gradient_bias_output));
			//中間層weight更新
			for(int j = 0; j < weight_hidden.length; j++) {
				for(int k = 0; k < weight_hidden[0].length; k++) {
					weight_hidden[j][k] -= this.eta * gradient_hidden[j][k];
				}
				bias_hidden[j] -= this.eta * gradient_bias_hidden[j]; 
			}
			//出力層weight更新
			for(int j = 0; j < weight_output.length; j++) {
				for(int k = 0; k < weight_output[0].length; k++) {
					weight_output[j][k] -= this.eta * gradient_output[j][k];
				}
				bias_output[j] -= this.eta * gradient_bias_output[j]; 
			}
			System.out.println("中間層更新weight" +Arrays.deepToString(weight_hidden));
			System.out.println("中間層バイアス更新" +Arrays.toString(bias_hidden));
			System.out.println("出力層更新weight" +Arrays.deepToString(weight_output));
			System.out.println("出力層バイアス更新" +Arrays.toString(bias_output));
		}
		
	}
	
	private double getInnerProduct(int[] a, double[] b) {
		// TODO Auto-generated method stub
		double answer = 0;
		for(int i = 0; i < a.length; i++) {
			answer += a[i] * b[i];
		}
		return answer;
	}

	public double getInnerProduct(double a[], double b[]) {
		double answer = 0;
		for(int i = 0; i < a.length; i++) {
			answer += a[i] * b[i];
		}
		return answer;
	}
	
	public double getSigmoid(double x) {
		int a = 1; //今回はa = 1
		return ( 1 / ( 1 + Math.exp(-a * x)));
	}
	
	public double[][] getTranspose(double [][]a){
		double t[][] = new double[a[0].length][a.length];
		for(int i = 0; i < a[0].length; i++) {
			for(int j = 0; j < a.length; j++) {
				t[i][j] = a[j][i];
			}
		}
		return t;
	}

}
