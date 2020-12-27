#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

float ReLU(float z){
	if(z > 0){
		return z;
	}
	else{
		return 0;
	}
}

float square_error(float output[2], float input[4], float y[2]){		//square error
	int i;
	float L = 0;
	float a[2] = {0, 1};
	float b[2] = {1, 0};
	
	if(input[0] > input[1]){										//左 
		for(i=0;i<2;i++){
			y[i] = a[i];
		}	
	} 
	else{															//右 
		for(i=0;i<2;i++){
			y[i] = b[i];
		}		
	} 
	//printf("y = {%f, %f}\n", y[0], y[1]); 
	for(i=0;i<2;i++){
		L = L + (output[i] - y[i])*(output[i] - y[i]);
	}																//計算square error
	return L;
}



int main(){
	int i, j, k, random;											//i, j, k為for迴圈用，random為取隨機值用，y[3]為實際值 
	float Loss, learning = 0.00001;										//Loss為誤差值，極為cross entropy計算後之結果，learning為learning rate 
	
	float initinput[2][4] = {										//全部的input 
		1, 0, 1, 0,
		0, 1, 0, 1
	};
	
	float y[2];
	float input[4];													//取隨機值得input之宣告 
	float array2[2];												//計算sigmoid的微分之宣告 
	
	float initweight1[2][4] = {										//input與第二層之間的weight之初始化 (zi=wi*xi+b) 
		1, 5, 4, 6,													//w11~w14，z1的weight 
		7, 3, 9, 8													//w21~w24，z2的weight 
	};
	float initweight2[2][2] = {										//第二層與output之間的weight之初始化 yi=wi*array[i]+b
		5, 7,														//w11~w12，array[0]的weight 
		6, 9,														//w21~w22，array[1]的weight												 
	};
	double initbias1[2] = {12, 23};									//b11~b12，z1的bias 
	double initbias2[3] = {17, 21};									//b21~b23，z2的bias
	
	srand(time(NULL));
	//start training↓↓↓↓
	for(k=1;k<=50000;k++){
		float array[2] = {0, 0};									//第二層layer之輸出array之初始化
		float output[2] = {0, 0};									//output之初始化 
		
		random = (rand()%2);
		//printf("random=%d\n", random);							//取隨機值 
		for(i=0;i<4;i++){
			input[i] = initinput[random][i];						//依據隨機值取一input 
		}
		
		for(i=0;i<2;i++){
			for(j=0;j<4;j++){
				array[i] = array[i] + initbias1[i];					//zi=wi*input[i]+bi
				//printf("(array+bias)array=%f\n", array[i]);
				array[i] = array[i] + initweight1[i][j]*input[j];	//矩陣相乘(wi*input[i]) 
				//printf("矩陣相乘array=%f\n", array[i]);
			}
		}
		for(i=0;i<2;i++){
			array[i] = ReLU(array[i]);								//array[i]=sigmiod(zi)
			//printf("(ReLU)array=%f\n", array[i]);
			if(array[i] > 0){
				array2[i] = 1;
			}
			else{
				array2[i] = 0;
			}
		}
		// 第二層↑↑↑↑
		for(i=0;i<2;i++){
			for(j=0;j<2;j++){
				output[i] = output[i] + initweight2[i][j]*array[j];	//矩陣相乘(wi*array[i])
			}
		}
		for(i=0;i<2;i++){
			output[i] = output[i] + initbias2[i];					//yi=wi*array[i]+bi
		}
		//printf("矩陣相乘output={%f, %f}\n", output[0], output[1]);
		Loss = square_error(output, input, y);						//square error計算Loss
		printf("epoch = %d, The Loss is %.10f\n", k, Loss);
		//backpropagation
		for(j=0;j<4;j++){
			initweight1[0][j] = initweight1[0][j] - learning*input[j]*array2[0]*(initweight2[0][0]*2*(output[0]-y[0]) + initweight2[1][0]*2*(output[1] - y[1]));
			//printf("w1%d=%f\n", j, initweight1[0][j]);
		}															//w1j之更新 
		for(j=0;j<4;j++){
			initweight1[1][j] = initweight1[1][j] - learning*input[j]*array2[1]*(initweight2[0][1]*2*(output[0]-y[0]) + initweight2[1][1]*2*(output[1] - y[1]));
			//printf("w2%d=%f\n", j, initweight1[1][j]);
		}															//w2j之更新 
		for(j=0;j<2;j++){
			initweight2[0][j] = initweight2[0][j] - learning*array[j]*2*(output[0]-y[0]);
			//printf("w3%d=%f\n", j, initweight2[0][j]);
		}															//w3j之更新 
		for(j=0;j<2;j++){
			initweight2[1][j] = initweight2[1][j] - learning*array[j]*2*(output[1]-y[1]);
			//printf("w4%d=%f\n", j, initweight2[1][j]);
		}															//w4j之更新 
		initbias1[0] = initbias1[0] - learning*array2[0]*(initweight2[0][0]*2*(output[0]-y[0]) + initweight2[1][0]*2*(output[1] - y[1]));
		//printf("bais11=%f\n", initbias1[0]);						//b11之更新	
		initbias1[1] = initbias1[1] - learning*array2[1]*(initweight2[0][1]*2*(output[0]-y[0]) + initweight2[1][1]*2*(output[1] - y[1]));
		//printf("bais12=%f\n", initbias1[1]);						//b12之更新	
		initbias2[0] = initbias2[0] - learning*2*(output[0]-y[0]);
		//printf("bais21=%f\n", initbias2[0]);						//b21之更新	
		initbias2[1] = initbias2[1] - learning*2*(output[1]-y[1]);
		//printf("bais22=%f\n", initbias2[1]);						//b22之更新	

		//system("pause");
	}
	//finish training↑↑↑↑
	//start testing↓↓↓↓
	float test[4];
	float a1, a2, a3, a4;
	while(1>0){
		float array[2] = {0, 0};									//第二層layer之輸出array之初始化
		float output[2] = {0, 0};									//output之初始化 
		printf("please key in \"1 0 1 0\" or \"0 1 0 1\" 2*2 array to test\n");
		scanf("%f%f%f%f", &a1, &a2, &a3, &a4);
		test[0] = a1;
		test[1] = a2;
		test[2] = a3;
		test[3] = a4;
		printf("The Array You Key In：\n{ %.0f, %.0f, \n  %.0f, %.0f }\n", test[0], test[1], test[2], test[3]);
		for(i=0;i<2;i++){
			for(j=0;j<4;j++){
				array[i] = array[i] + initbias1[i];					//zi=wi*input[i]+bi
				//printf("(array+bias)array=%f\n", array[i]);
				array[i] = array[i] + initweight1[i][j]*test[j];	//矩陣相乘(wi*input[i]) 
				//printf("矩陣相乘array=%f\n", array[i]);
			}
		}
		for(i=0;i<2;i++){
			array[i] = ReLU(array[i]);								//array[i]=sigmiod(zi)
			//printf("(ReLU)array=%f\n", array[i]);
			if(array[i] > 0){
				array2[i] = 1;
			}
			else{
				array2[i] = 0;
			}
		}
		// 第二層↑↑↑↑
		for(i=0;i<2;i++){
			for(j=0;j<2;j++){
				output[i] = output[i] + initweight2[i][j]*array[j];	//矩陣相乘(wi*array[i])
			}
		}
		for(i=0;i<2;i++){
			output[i] = output[i] + initbias2[i];					//yi=wi*array[i]+bi
		}

		Loss = square_error(output, test, y);						//square error計算Loss
		printf("預測值 = {%f, %f}, 實際值 = {%.0f, %.0f}\n", output[0], output[1], y[0], y[1]);
		printf("The Loss is %.10f\n", Loss);
		
		char b[2];
		char c[2] = "y";
		printf("do you want to test again?(y/n)");
		scanf("%s", b);
		if(strcmp(b, c) != 0){
			break;
		}
	}
	

	system("pause");
	return 0;
}
