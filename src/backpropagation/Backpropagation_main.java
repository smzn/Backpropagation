package backpropagation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Backpropagation_main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int data_number = 64, input_category = 12, teacher_category = 2;
		int input [][] = new int[data_number][input_category];
		int teacher[][] = new int[data_number][teacher_category];
		int hidden_node = 3; 
		
		Backpropagation_main bmain = new Backpropagation_main();
		bmain.getCSV2("csv/neuraldata.csv", data_number, input_category + teacher_category, input, teacher);
		System.out.println("Input Data" +Arrays.deepToString(input));
		System.out.println("Teacher Data" +Arrays.deepToString(teacher));
		
		Backpropagation_lib blib = new Backpropagation_lib(hidden_node, 3, input, teacher);
		blib.getNetwork();
	}

	//複数種類のデータを一度に取り込む場合
	public void getCSV2(String path, int row, int column, int[][] input, int[][] teacher ) {
		//CSVから取り込み
		try {
			File f = new File(path);
			BufferedReader br = new BufferedReader(new FileReader(f));
					 
			String[][] data = new String[row][column]; 
			String line = br.readLine();
			for (int i = 0; line != null; i++) {
				data[i] = line.split(",", 0);
				line = br.readLine();
			}
			br.close();
			
			// CSVから読み込んだ配列の中身を処理
			for(int i = 0; i < data.length; i++) {
				for(int j = 0; j < data[0].length; j++) {
					if( j < data[0].length -2 ) {
						input[i][j] = Integer.parseInt(data[i][j]);
					} else teacher[i][j - (data[0].length - 2) ] = Integer.parseInt(data[i][j]);
				}
			} 

		} catch (IOException e) {
				System.out.println(e);
		}
			//CSVから取り込みここまで	
	}
	
}
