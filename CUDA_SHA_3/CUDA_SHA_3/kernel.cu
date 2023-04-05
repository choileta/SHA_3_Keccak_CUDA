#define CUDA_API_PER_THREAD_DEFAULT_STEAM
#include "SHA3.cuh"


__device__ __inline__ void Absorb(u64* state)
{
	asm("{\n\t"
		//register Setting
		".reg.b64			temp;		\n\t"
		".reg.b64			buf0;		\n\t"
		".reg.b64			buf1;		\n\t"

		".reg.b64			theta0;		\n\t"
		".reg.b64			theta1;		\n\t"
		".reg.b64			theta2;		\n\t"
		".reg.b64			theta3;		\n\t"
		".reg.b64			theta4;		\n\t"
		".reg.b64			pi0;		\n\t"
		".reg.b64			pi1;		\n\t"
		".reg.b64			pi2;		\n\t"

		".reg.b64			lane00;		\n\t"
		".reg.b64			lane01;		\n\t"
		".reg.b64			lane02;		\n\t"
		".reg.b64			lane03;		\n\t"
		".reg.b64			lane04;		\n\t"
		".reg.b64			lane05;		\n\t"
		".reg.b64			lane06;		\n\t"
		".reg.b64			lane07;		\n\t"
		".reg.b64			lane08;		\n\t"
		".reg.b64			lane09;		\n\t"
		".reg.b64			lane10;		\n\t"
		".reg.b64			lane11;		\n\t"
		".reg.b64			lane12;		\n\t"
		".reg.b64			lane13;		\n\t"
		".reg.b64			lane14;		\n\t"
		".reg.b64			lane15;		\n\t"
		".reg.b64			lane16;		\n\t"
		".reg.b64			lane17;		\n\t"
		".reg.b64			lane18;		\n\t"
		".reg.b64			lane19;		\n\t"
		".reg.b64			lane20;		\n\t"
		".reg.b64			lane21;		\n\t"
		".reg.b64			lane22;		\n\t"
		".reg.b64			lane23;		\n\t"
		".reg.b64			lane24;		\n\t"

		"mov.b64			lane00, %25;	\n\t"
		"mov.b64			lane01, %26;	\n\t"
		"mov.b64			lane02, %27;	\n\t"
		"mov.b64			lane03, %28;	\n\t"
		"mov.b64			lane04, %29;	\n\t"
		"mov.b64			lane05, %30;	\n\t"
		"mov.b64			lane06, %31;	\n\t"
		"mov.b64			lane07, %32;	\n\t"
		"mov.b64			lane08, %33;	\n\t"
		"mov.b64			lane09, %34;	\n\t"
		"mov.b64			lane10, %35;	\n\t"
		"mov.b64			lane11, %36;	\n\t"
		"mov.b64			lane12, %37;	\n\t"
		"mov.b64			lane13, %38;	\n\t"
		"mov.b64			lane14, %39;	\n\t"
		"mov.b64			lane15, %40;	\n\t"
		"mov.b64			lane16, %41;	\n\t"
		"mov.b64			lane17, %42;	\n\t"
		"mov.b64			lane18, %43;	\n\t"
		"mov.b64			lane19, %44;	\n\t"
		"mov.b64			lane20, %45;	\n\t"
		"mov.b64			lane21, %46;	\n\t"
		"mov.b64			lane22, %47;	\n\t"
		"mov.b64			lane23, %48;	\n\t"
		"mov.b64			lane24, %49;	\n\t"

		//0Round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000000100000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//1round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000808200000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//2round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000808a80000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//3round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000800080000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//4round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000808b00000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		"mov.b64			%0, lane00;						\n\t"
		"mov.b64			%1, lane01;						\n\t"
		"mov.b64			%2, lane02;						\n\t"
		"mov.b64			%3, lane03;						\n\t"
		"mov.b64			%4, lane04;						\n\t"

		//5Round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000000100000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//6round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000808180000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//7round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000800980000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//8round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000008a00000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//9round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000008800000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		"mov.b64			%0, lane00;						\n\t"
		"mov.b64			%1, lane01;						\n\t"
		"mov.b64			%2, lane02;						\n\t"
		"mov.b64			%3, lane03;						\n\t"
		"mov.b64			%4, lane04;						\n\t"

		//10Round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000800900000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//11round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000000a00000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//12round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000808b00000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//13round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000008b80000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//14round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000808980000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		"mov.b64			%0, lane00;						\n\t"
		"mov.b64			%1, lane01;						\n\t"
		"mov.b64			%2, lane02;						\n\t"
		"mov.b64			%3, lane03;						\n\t"
		"mov.b64			%4, lane04;						\n\t"

		//15Round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000800380000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//16round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000800280000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//17round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000008080000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//18round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000800a00000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//19round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000000a80000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		"mov.b64			%0, lane00;						\n\t"
		"mov.b64			%1, lane01;						\n\t"
		"mov.b64			%2, lane02;						\n\t"
		"mov.b64			%3, lane03;						\n\t"
		"mov.b64			%4, lane04;						\n\t"

		//20Round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000808180000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//21round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x0000808080000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"
		//////////////////////////////////////////////////////////
		//22round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000000100000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		//23round
		//initial Theta
		"xor.b64			temp,		lane00,		lane05;	\n\t"
		"xor.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			temp,		temp,		lane15;	\n\t"
		"xor.b64			theta0,		temp,		lane20;	\n\t"

		"xor.b64			temp,		lane01,		lane06; \n\t"
		"xor.b64			temp,		temp,		lane11; \n\t"
		"xor.b64			temp,		temp,		lane16; \n\t"
		"xor.b64			theta1,		temp,		lane21; \n\t"

		"xor.b64			temp,		lane02,		lane07;	\n\t"
		"xor.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			theta2,		temp,		lane22;	\n\t"

		"xor.b64			temp,		lane03,		lane08; \n\t"
		"xor.b64			temp,		temp,		lane13; \n\t"
		"xor.b64			temp,		temp,		lane18; \n\t"
		"xor.b64			theta3,		temp,		lane23; \n\t"

		"xor.b64			temp,		lane04,		lane09;	\n\t"
		"xor.b64			temp,		temp,		lane14;	\n\t"
		"xor.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			theta4,		temp,		lane24;	\n\t"

		//theta process & rho process
		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta1,		1;		\n\t"
		"shr.b64			buf0,		theta1,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta4;\n\t"

		//state[0]
		"xor.b64			lane00,		lane00,		temp;	\n\t"

		//state[5]
		"xor.b64			lane05,		lane05,		temp;	\n\t"
		"shl.b64			buf0,		lane05,		36;		\n\t"
		"shr.b64			buf1,		lane05,		28;		\n\t"
		"or.b64				lane05,		buf0,		buf1;	\n\t"

		//state[10]
		"xor.b64			lane10,		lane10,		temp;	\n\t"
		"shl.b64			buf0,		lane10,		3;		\n\t"
		"shr.b64			buf1,		lane10,		61;		\n\t"
		"or.b64				lane10,		buf0,		buf1;	\n\t"

		//state[15]
		"xor.b64			lane15,		lane15,		temp;	\n\t"
		"shl.b64			buf0,		lane15,		41;		\n\t"
		"shr.b64			buf1,		lane15,		23;		\n\t"
		"or.b64				lane15,		buf0,		buf1;	\n\t"

		//state[20]
		"xor.b64			lane20,		lane20,		temp;	\n\t"
		"shl.b64			buf0,		lane20,		18;		\n\t"
		"shr.b64			buf1,		lane20,		46;		\n\t"
		"or.b64				lane20,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta2,		1;		\n\t"
		"shr.b64			buf0,		theta2,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta0;	\n\t"

		//state[1]
		"xor.b64			lane01,		lane01,		temp;	\n\t"
		"shl.b64			buf0,		lane01,		1;		\n\t"
		"shr.b64			buf1,		lane01,		63;		\n\t"
		"or.b64				lane01,		buf0,		buf1;	\n\t"

		//state[6]
		"xor.b64			lane06,		lane06,		temp;	\n\t"
		"shl.b64			buf0,		lane06,		44;		\n\t"
		"shr.b64			buf1,		lane06,		20;		\n\t"
		"or.b64				lane06,		buf0,		buf1;	\n\t"

		//state[11]
		"xor.b64			lane11,		lane11,		temp;	\n\t"
		"shl.b64			buf0,		lane11,		10;		\n\t"
		"shr.b64			buf1,		lane11,		54;		\n\t"
		"or.b64				lane11,		buf0,		buf1;	\n\t"

		//state[16]
		"xor.b64			lane16,		lane16,		temp;	\n\t"
		"shl.b64			buf0,		lane16,		45;		\n\t"
		"shr.b64			buf1,		lane16,		19;		\n\t"
		"or.b64				lane16,		buf0,		buf1;	\n\t"

		//state[21]
		"xor.b64			lane21,		lane21,		temp;	\n\t"
		"shl.b64			buf0,		lane21,		2;		\n\t"
		"shr.b64			buf1,		lane21,		62;		\n\t"
		"or.b64				lane21,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta3,		1;		\n\t"
		"shr.b64			buf0,		theta3,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta1;	\n\t"

		//state[2]
		"xor.b64			lane02,		lane02,		temp;	\n\t"
		"shl.b64			buf0,		lane02,		62;		\n\t"
		"shr.b64			buf1,		lane02,		2;		\n\t"
		"or.b64				lane02,		buf0,		buf1;	\n\t"

		//state[7]
		"xor.b64			lane07,		lane07,		temp;	\n\t"
		"shl.b64			buf0,		lane07,		6;		\n\t"
		"shr.b64			buf1,		lane07,		58;		\n\t"
		"or.b64				lane07,		buf0,		buf1;	\n\t"

		//state[12]
		"xor.b64			lane12,		lane12,		temp;	\n\t"
		"shl.b64			buf0,		lane12,		43;		\n\t"
		"shr.b64			buf1,		lane12,		21;		\n\t"
		"or.b64				lane12,		buf0,		buf1;	\n\t"

		//state[17]
		"xor.b64			lane17,		lane17,		temp;	\n\t"
		"shl.b64			buf0,		lane17,		15;		\n\t"
		"shr.b64			buf1,		lane17,		49;		\n\t"
		"or.b64				lane17,		buf0,		buf1;	\n\t"

		//state[22]
		"xor.b64			lane22,		lane22,		temp;	\n\t"
		"shl.b64			buf0,		lane22,		61;		\n\t"
		"shr.b64			buf1,		lane22,		3;		\n\t"
		"or.b64				lane22,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta4,		1;		\n\t"
		"shr.b64			buf0,		theta4,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta2;	\n\t"

		//state[3]
		"xor.b64			lane03,		lane03,		temp;	\n\t"
		"shl.b64			buf0,		lane03,		28;		\n\t"
		"shr.b64			buf1,		lane03,		36;		\n\t"
		"or.b64				lane03,		buf0,		buf1;	\n\t"

		//state[8]
		"xor.b64			lane08,		lane08,		temp;	\n\t"
		"shl.b64			buf0,		lane08,		55;		\n\t"
		"shr.b64			buf1,		lane08,		9;		\n\t"
		"or.b64				lane08,		buf0,		buf1;	\n\t"

		//state[13]
		"xor.b64			lane13,		lane13,		temp;	\n\t"
		"shl.b64			buf0,		lane13,		25;		\n\t"
		"shr.b64			buf1,		lane13,		39;		\n\t"
		"or.b64				lane13,		buf0,		buf1;	\n\t"

		//state[18]
		"xor.b64			lane18,		lane18,		temp;	\n\t"
		"shl.b64			buf0,		lane18,		21;		\n\t"
		"shr.b64			buf1,		lane18,		43;		\n\t"
		"or.b64				lane18,		buf0,		buf1;	\n\t"

		//state[23]
		"xor.b64			lane23,		lane23,		temp;	\n\t"
		"shl.b64			buf0,		lane23,		56;		\n\t"
		"shr.b64			buf1,		lane23,		8;		\n\t"
		"or.b64				lane23,		buf0,		buf1;	\n\t"

		/////////////////////////////////////////////////////////
		"shl.b64			temp,		theta0,		1;		\n\t"
		"shr.b64			buf0,		theta0,		63;		\n\t"
		"or.b64				temp,		temp,		buf0;	\n\t"
		"xor.b64			temp,		temp,		theta3;	\n\t"

		//state[4]
		"xor.b64			lane04,		lane04,		temp;	\n\t"
		"shl.b64			buf0,		lane04,		27;		\n\t"
		"shr.b64			buf1,		lane04,		37;		\n\t"
		"or.b64				lane04,		buf0,		buf1;	\n\t"

		//state[9]
		"xor.b64			lane09,		lane09,		temp;	\n\t"
		"shl.b64			buf0,		lane09,		20;		\n\t"
		"shr.b64			buf1,		lane09,		44;		\n\t"
		"or.b64				lane09,		buf0,		buf1;	\n\t"

		//state[14]
		"xor.b64			lane14,		lane14,		temp;	\n\t"
		"shl.b64			buf0,		lane14,		39;		\n\t"
		"shr.b64			buf1,		lane14,		25;		\n\t"
		"or.b64				lane14,		buf0,		buf1;	\n\t"

		//state[19]
		"xor.b64			lane19,		lane19,		temp;	\n\t"
		"shl.b64			buf0,		lane19,		8;		\n\t"
		"shr.b64			buf1,		lane19,		56;		\n\t"
		"or.b64				lane19,		buf0,		buf1;	\n\t"

		//state[24]
		"xor.b64			lane24,		lane24,		temp;	\n\t"
		"shl.b64			buf0,		lane24,		14;		\n\t"
		"shr.b64			buf1,		lane24,		50;		\n\t"
		"or.b64				lane24,		buf0,		buf1;	\n\t"

		//Chi & iota Process
		//state[0] update
		"mov.b64			theta0,		lane00;				\n\t"
		"not.b64			temp,		lane06;				\n\t"
		"and.b64			temp,		temp,		lane12;	\n\t"
		"xor.b64			temp,		temp,		lane00;	\n\t"
		"xor.b64			lane00,		temp,		0x8000800880000000;	\n\t"

		//state[1] update
		"mov.b64			theta1,		lane01;				\n\t"
		"not.b64			temp,		lane12;				\n\t"
		"and.b64			temp,		temp,		lane18;	\n\t"
		"xor.b64			lane01,		temp,		lane06;	\n\t"

		//state[2] update
		"mov.b64			theta2,		lane02;				\n\t"
		"not.b64			temp,		lane18;				\n\t"
		"and.b64			temp,		temp,		lane24;	\n\t"
		"xor.b64			lane02,		temp,		lane12;	\n\t"

		//state[3] update
		"mov.b64			theta3,		lane03;				\n\t"
		"not.b64			temp,		lane24;				\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane03,		temp,		lane18;	\n\t"

		//state[4] update
		"mov.b64			theta4,		lane04;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		lane06;	\n\t"
		"xor.b64			lane04,		temp,		lane24;	\n\t"

		//////////////////////////////////////////////////////////
		//theta0 -> X
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> lane03
		//theta4 -> lane04

		//state[5] update
		"mov.b64			theta0,		lane05;				\n\t"
		"not.b64			temp,		lane09;				\n\t"
		"and.b64			temp,		temp,		lane10;	\n\t"
		"xor.b64			lane05,		temp,		theta3;	\n\t"
		//theta0 -> lane05

		//state[6] update
		"not.b64			temp,		lane10;				\n\t"
		"and.b64			temp,		temp,		lane16;	\n\t"
		"xor.b64			lane06,		temp,		lane09;	\n\t"

		//state[7] update
		"mov.b64			pi1,		lane07;				\n\t"
		"not.b64			temp,		lane16;				\n\t"
		"and.b64			temp,		temp,		lane22;	\n\t"
		"xor.b64			lane07,		temp,		lane10;	\n\t"
		//pi1 -> lane07

		//state[8] update
		"mov.b64			pi2,		lane08;				\n\t"
		"not.b64			temp,		lane22;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane08,		temp,		lane16;	\n\t"

		//state[9] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane09;	\n\t"
		"xor.b64			lane09,		temp,		lane22;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane01
		//theta2 -> lane02
		//theta3 -> X
		//theta4 -> lane04
		//pi0 -> X
		//pi1 -> lane07
		//pi2 -> lane08

		//state[10] update
		"not.b64			temp,		pi1;				\n\t"
		"and.b64			temp,		temp,		lane13;	\n\t"
		"xor.b64			lane10,		temp,		theta1;	\n\t"

		//state[11] update
		"mov.b64			theta3,		lane11;				\n\t"
		"not.b64			temp,		lane13;				\n\t"
		"and.b64			temp,		temp,		lane19;	\n\t"
		"xor.b64			lane11,		temp,		pi1;	\n\t"
		//theta03 -> lane 11

		//state[12] update
		"not.b64			temp,		lane19;				\n\t"
		"and.b64			temp,		temp,		lane20;	\n\t"
		"xor.b64			lane12,		temp,		lane13;	\n\t"

		//state[13] update
		"not.b64			temp,		lane20;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane13,		temp,		lane19;	\n\t"

		//state[14] update
		"mov.b64			pi0,		lane14;				\n\t"
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		pi1;	\n\t"
		"xor.b64			lane14,		temp,		lane20;	\n\t"
		//////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> X
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[15] update
		"mov.b64			theta1,		lane15;				\n\t"
		"not.b64			temp,		theta0;				\n\t"
		"and.b64			temp,		temp,		theta3;	\n\t"
		"xor.b64			lane15,		temp,		theta4;	\n\t"
		//theta1 -> lane15

		//state[16] update
		"not.b64			temp,		theta3;				\n\t"
		"and.b64			temp,		temp,		lane17;	\n\t"
		"xor.b64			lane16,		temp,		theta0;	\n\t"

		//state[18] update
		"not.b64			temp,		lane23;				\n\t"
		"and.b64			temp,		temp,		theta4;	\n\t"
		"xor.b64			lane18,		temp,		lane17;	\n\t"

		//state[17] update
		"not.b64			temp,		lane17;				\n\t"
		"and.b64			temp,		temp,		lane23;	\n\t"
		"xor.b64			lane17,		temp,		theta3;	\n\t"

		//state[19] update
		"not.b64			temp,		theta4;			\n\t"
		"and.b64			temp,		temp,		theta0;	\n\t"
		"xor.b64			lane19,		temp,		lane23;	\n\t"
		////////////////////////////////////////////////////////
		//theta0 -> lane05
		//theta1 -> lane15
		//theta2 -> lane02
		//theta3 -> lane11
		//theta4 -> lane04
		//pi0 -> lane14
		//pi1 -> X
		//pi2 -> lane08

		//state[20] update
		"not.b64			temp,		pi2;				\n\t"
		"and.b64			temp,		temp,		pi0;	\n\t"
		"xor.b64			lane20,		temp,		theta2;	\n\t"

		//state[22] update
		"not.b64			temp,		theta1;				\n\t"
		"and.b64			temp,		temp,		lane21;	\n\t"
		"xor.b64			lane22,		temp,		pi0;	\n\t"

		//state[23] update
		"not.b64			temp,		lane21;				\n\t"
		"and.b64			temp,		temp,		theta2;	\n\t"
		"xor.b64			lane23,		temp,		theta1;	\n\t"

		//state[24] update
		"not.b64			temp,		theta2;				\n\t"
		"and.b64			temp,		temp,		pi2;	\n\t"
		"xor.b64			lane24,		temp,		lane21;	\n\t"

		//state[21] update
		"not.b64			temp,		pi0;				\n\t"
		"and.b64			temp,		temp,		theta1;	\n\t"
		"xor.b64			lane21,		temp,		pi2;	\n\t"

		"mov.b64			%0, lane00;						\n\t"
		"mov.b64			%1, lane01;						\n\t"
		"mov.b64			%2, lane02;						\n\t"
		"mov.b64			%3, lane03;						\n\t"
		"mov.b64			%4, lane04;						\n\t"

		"mov.b64			%5, lane05;						\n\t"
		"mov.b64			%6, lane06;						\n\t"
		"mov.b64			%7, lane07;						\n\t"
		"mov.b64			%8, lane08;						\n\t"
		"mov.b64			%9, lane09;						\n\t"

		"mov.b64			%10, lane10;						\n\t"
		"mov.b64			%11, lane11;						\n\t"
		"mov.b64			%12, lane12;						\n\t"
		"mov.b64			%13, lane13;						\n\t"
		"mov.b64			%14, lane14;						\n\t"

		"mov.b64			%15, lane15;						\n\t"
		"mov.b64			%16, lane16;						\n\t"
		"mov.b64			%17, lane17;						\n\t"
		"mov.b64			%18, lane18;						\n\t"
		"mov.b64			%19, lane19;						\n\t"

		"mov.b64			%20, lane20;						\n\t"
		"mov.b64			%21, lane21;						\n\t"
		"mov.b64			%22, lane22;						\n\t"
		"mov.b64			%23, lane23;						\n\t"
		"mov.b64			%24, lane24;						}\n\t"

		: "=l"(state[0]), "=l"(state[1]), "=l"(state[2]), "=l"(state[3]), "=l"(state[4]),
		"=l"(state[5]), "=l"(state[6]), "=l"(state[7]), "=l"(state[8]), "=l"(state[9]),
		"=l"(state[10]), "=l"(state[11]), "=l"(state[12]), "=l"(state[13]), "=l"(state[14]),
		"=l"(state[15]), "=l"(state[16]), "=l"(state[17]), "=l"(state[18]), "=l"(state[19]),
		"=l"(state[20]), "=l"(state[21]), "=l"(state[22]), "=l"(state[23]), "=l"(state[24])

		: "l"(state[0]), "l"(state[1]), "l"(state[2]), "l"(state[3]), "l"(state[4]),
		"l"(state[5]), "l"(state[6]), "l"(state[7]), "l"(state[8]), "l"(state[9]),
		"l"(state[10]), "l"(state[11]), "l"(state[12]), "l"(state[13]), "l"(state[14]),
		"l"(state[15]), "l"(state[16]), "l"(state[17]), "l"(state[18]), "l"(state[19]),
		"l"(state[20]), "l"(state[21]), "l"(state[22]), "l"(state[23]), "l"(state[24])
	);
}

__global__ void Keccak256(u8* in, u8* out)
{
	u8 msgtemp[136];
	msgtemp[1] = 0x06;
	msgtemp[135] = 0x80;
	u64 state[25] = { 0, };
	//int index = 32 * (blockDim.x * blockIdx.x) + (32 * threadIdx.x);
	int index = (blockDim.x * blockIdx.x) + (threadIdx.x);
	int dist = blockDim.x * gridDim.x;
	msgtemp[0] = in[index];
	for (int i = 0; i < 17; i++)
		state[i] = ENDIAN_CHANGE(((u64*)msgtemp)[i]);

	Absorb(state);

	for (int i = 0; i < 25; i++)
		state[i] = ENDIAN_CHANGE(state[i]);

	for (int i = 0; i < 4; i++) {
		out[index + (dist * (8 * i))] = state[i] & 0xff;
		out[index + ((dist) * (8 * i + 1))] = (state[i] >> 8) & 0xff;
		out[index + ((dist) * (8 * i + 2))] = (state[i] >> 16) & 0xff;
		out[index + ((dist) * (8 * i + 3))] = (state[i] >> 24) & 0xff;
		out[index + ((dist) * (8 * i + 4))] = (state[i] >> 32) & 0xff;
		out[index + ((dist) * (8 * i + 5))] = (state[i] >> 40) & 0xff;
		out[index + ((dist) * (8 * i + 6))] = (state[i] >> 48) & 0xff;
		out[index + ((dist) * (8 * i + 7))] = (state[i] >> 56) & 0xff;
	}
}

__global__ void Keccak512(u8* in, u8* out)
{
	u8 msgtemp[72];
	msgtemp[1] = 0x06;
	msgtemp[71] = 0x80;
	u64 state[25] = { 0, };
	//int index = 64 * (blockDim.x * blockIdx.x) + (64 * threadIdx.x);
	int index = (blockDim.x * blockIdx.x) + (threadIdx.x);
	msgtemp[0] = in[index];
	int dist = blockDim.x * gridDim.x;
	for (int i = 0; i < 9; i++)
		state[i] = ENDIAN_CHANGE(((u64*)msgtemp)[i]);

	Absorb(state);
	for (int i = 0; i < 25; i++)
		state[i] = ENDIAN_CHANGE(state[i]);

	/*for (int i = 0; i < 8; i++) {
		out[index + (8 * i)] = state[i] & 0xff;
		out[index + (8 * i) + 1] = (state[i] >> 8) & 0xff;
		out[index + (8 * i) + 2] = (state[i] >> 16) & 0xff;
		out[index + (8 * i) + 3] = (state[i] >> 24) & 0xff;
		out[index + (8 * i) + 4] = (state[i] >> 32) & 0xff;
		out[index + (8 * i) + 5] = (state[i] >> 40) & 0xff;
		out[index + (8 * i) + 6] = (state[i] >> 48) & 0xff;
		out[index + (8 * i) + 7] = (state[i] >> 56) & 0xff;
	}*/
	for (int i = 0; i < 8; i++) {
		out[index + (dist * (8 * i))] = state[i] & 0xff;
		out[index + ((dist) * (8 * i + 1))] = (state[i] >> 8) & 0xff;
		out[index + ((dist) * (8 * i + 2))] = (state[i] >> 16) & 0xff;
		out[index + ((dist) * (8 * i + 3))] = (state[i] >> 24) & 0xff;
		out[index + ((dist) * (8 * i + 4))] = (state[i] >> 32) & 0xff;
		out[index + ((dist) * (8 * i + 5))] = (state[i] >> 40) & 0xff;
		out[index + ((dist) * (8 * i + 6))] = (state[i] >> 48) & 0xff;
		out[index + ((dist) * (8 * i + 7))] = (state[i] >> 56) & 0xff;
	}
}

u8 H_in[16384 * 512 * 3];

void SHA3_256_Func(int Block_size, int Thread_size)
{
	cudaEvent_t start, stop;
	float elapsed_time_ms = 0.0f;
	u8* D_out = NULL;
	u8* D_in = NULL;
	cudaMalloc((void**)&D_out, Block_size * Thread_size * 32);
	cudaMalloc((void**)&D_in, Block_size * Thread_size);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; i++) {
		cudaMemcpy(D_in, H_in, Block_size * Thread_size, cudaMemcpyHostToDevice);
		Keccak256 << <Block_size, Thread_size >> > (D_in, D_out);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms = elapsed_time_ms / 100;
	elapsed_time_ms = (Block_size * Thread_size * 136) / elapsed_time_ms;
	elapsed_time_ms *= 1000;
	elapsed_time_ms /= (1024 * 1024 * 1024);


	printf("Grid : %d, Block : %d, Performance : %4.2f GB/s\n", Block_size, Thread_size, elapsed_time_ms);
	cudaFree(D_in);
	cudaFree(D_out);
}

void SHA3_512_Func(int Block_size, int Thread_size)
{
	cudaEvent_t start, stop;
	float elapsed_time_ms = 0.0f;
	u8* D_out = NULL;
	u8* D_in = NULL;
	cudaMalloc((void**)&D_in, Block_size * Thread_size);
	cudaMalloc((void**)&D_out, Block_size * Thread_size * 64);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; i++) {
		cudaMemcpy(D_in, H_in, Block_size * Thread_size, cudaMemcpyHostToDevice);
		Keccak512 << <Block_size, Thread_size >> > (D_in, D_out);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms = elapsed_time_ms / 100;
	elapsed_time_ms = (Block_size * Thread_size * 72) / elapsed_time_ms;
	elapsed_time_ms *= 1000;
	elapsed_time_ms /= (1024 * 1024 * 1024);
	printf("Grid : %d, Block : %d, Performance : %4.2f GB/s\n", Block_size, Thread_size, elapsed_time_ms);
	cudaFree(D_in);
	cudaFree(D_out);
}

void SHA3_256_stream(int Block_size, int Thread_size) {
	cudaEvent_t start, stop;
	cudaStream_t stream[3];
	float elapsed_time_ms = 0.0f;
	u8* streamout0 = NULL;
	u8* streamout1 = NULL;
	u8* streamout2 = NULL;
	u8* streamin0 = NULL;
	u8* streamin1 = NULL;
	u8* streamin2 = NULL;


	cudaMalloc((void**)&streamin0, Block_size * Thread_size);
	cudaMalloc((void**)&streamin1, Block_size * Thread_size);
	cudaMalloc((void**)&streamin2, Block_size * Thread_size);

	cudaMalloc((void**)&streamout0, Block_size * Thread_size * 32);
	cudaMalloc((void**)&streamout1, Block_size * Thread_size * 32);
	cudaMalloc((void**)&streamout2, Block_size * Thread_size * 32);

	for (int i = 0; i < 3; i++)
		cudaStreamCreate(&stream[i]);



	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 10000; i++)
	{
		cudaMemcpyAsync(streamin0, H_in, Block_size * Thread_size, cudaMemcpyHostToDevice, stream[0]);
		Keccak256 << <Block_size, Thread_size, 0, stream[0] >> > (streamin0, streamout0);
		cudaMemcpyAsync(streamin1, H_in + (Block_size * Thread_size), Block_size * Thread_size, cudaMemcpyHostToDevice, stream[1]);
		Keccak256 << <Block_size, Thread_size, 0, stream[1] >> > (streamin1, streamout1);
		cudaMemcpyAsync(streamin2, H_in + (Block_size * Thread_size * 2), Block_size * Thread_size, cudaMemcpyHostToDevice, stream[2]);
		Keccak256 << <Block_size, Thread_size, 0, stream[2] >> > (streamin2, streamout2);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms = elapsed_time_ms / 30000;
	elapsed_time_ms = (Block_size * Thread_size * 136 * 3) / elapsed_time_ms;
	elapsed_time_ms *= 1000;
	elapsed_time_ms /= (1024 * 1024 * 1024);
	printf("Grid : %d, Block : %d, Performance : %4.2f GB/s\n", Block_size, Thread_size, elapsed_time_ms);

	cudaFree(streamout0);
	cudaFree(streamout1);
	cudaFree(streamout2);
	cudaFree(streamin0);
	cudaFree(streamin1);
	cudaFree(streamin2);

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);

}

void SHA3_512_stream(int Block_size, int Thread_size) {
	cudaEvent_t start, stop;
	cudaStream_t stream[3];
	float elapsed_time_ms = 0.0f;
	u8* streamout0 = NULL;
	u8* streamout1 = NULL;
	u8* streamout2 = NULL;
	u8* streamin0 = NULL;
	u8* streamin1 = NULL;
	u8* streamin2 = NULL;


	cudaMalloc((void**)&streamin0, Block_size * Thread_size);
	cudaMalloc((void**)&streamin1, Block_size * Thread_size);
	cudaMalloc((void**)&streamin2, Block_size * Thread_size);

	cudaMalloc((void**)&streamout0, Block_size * Thread_size * 64);
	cudaMalloc((void**)&streamout1, Block_size * Thread_size * 64);
	cudaMalloc((void**)&streamout2, Block_size * Thread_size * 64);

	for (int i = 0; i < 3; i++)
		cudaStreamCreate(&stream[i]);




	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 10000; i++)
	{
		cudaMemcpyAsync(streamin0, H_in, Block_size * Thread_size, cudaMemcpyHostToDevice, stream[0]);
		Keccak512 << <Block_size, Thread_size, 0, stream[0] >> > (streamin0, streamout0);
		cudaMemcpyAsync(streamin1, H_in + (Block_size * Thread_size), Block_size * Thread_size, cudaMemcpyHostToDevice, stream[1]);
		Keccak512 << <Block_size, Thread_size, 0, stream[1] >> > (streamin1, streamout1);
		cudaMemcpyAsync(streamin2, H_in + (Block_size * Thread_size * 2), Block_size * Thread_size, cudaMemcpyHostToDevice, stream[2]);
		Keccak512 << <Block_size, Thread_size, 0, stream[2] >> > (streamin2, streamout2);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms = elapsed_time_ms / 30000;
	elapsed_time_ms = (Block_size * Thread_size * 72 * 3) / elapsed_time_ms;
	elapsed_time_ms *= 1000;
	elapsed_time_ms /= (1024 * 1024 * 1024);
	printf("Grid : %d, Block : %d, Performance : %4.2f GB/s\n", Block_size, Thread_size, elapsed_time_ms);

	cudaFree(streamout0);
	cudaFree(streamout1);
	cudaFree(streamout2);
	cudaFree(streamin0);
	cudaFree(streamin1);
	cudaFree(streamin2);

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);
}

int main()
{
	for (int i = 0; i < 16384 * 512 * 3; i++) {
		H_in[i] = i + 1;
	}
	printf("SHA3-256 Testing\n");
	//SHA3_256_Func(1024, 64);
	//SHA3_256_Func(1024, 128);
	//SHA3_256_Func(2048, 128);
	//SHA3_256_Func(4096, 128);
	//SHA3_256_Func(8192, 128);
	//SHA3_256_Func(16384, 128);
	SHA3_256_Func(16384, 256);
	SHA3_256_Func(16384, 512);
	printf("SHA3-256 END\n");
	printf("SHA3-512 Testing\n");
	//SHA3_512_Func(1024, 64);
	//SHA3_512_Func(1024, 128);
	//SHA3_512_Func(2048, 128);
	//SHA3_512_Func(4096, 128);
	//SHA3_512_Func(8192, 128);
	//SHA3_512_Func(16384, 128);
	SHA3_512_Func(16384, 256);
	SHA3_512_Func(16384, 512);
	printf("SHA3-512 END\n");
	printf("SHA3-256 cuda Stream Testing\n");
	//SHA3_256_stream(256, 64);
	//SHA3_256_stream(512, 64);
	//SHA3_256_stream(1024, 64);
	//SHA3_256_stream(1024, 128);
	//SHA3_256_stream(2048, 128);
	//SHA3_256_stream(4096, 128);
	//SHA3_256_stream(8192, 128);
	//SHA3_256_stream(16384, 128);
	printf("SHA3-256 cuda Stream END\n");
	printf("SHA3-512 cuda Stream Testing\n");
	//SHA3_512_stream(1024, 64);
	//SHA3_512_stream(1024, 128);
	//SHA3_512_stream(2048, 128);
	//SHA3_512_stream(4096, 128);
	//SHA3_512_stream(8192, 128);
	//SHA3_512_stream(16384, 128);
	printf("SHA3-256 cuda Stream END\n");
}