#include <WinSock2.h>
#include <windows.h>
#include <shlwapi.h>
#include <fstream>

#pragma  comment(lib,"ws2_32.lib")

// #define SWRS_USES_HASH
#include "swrs.h"

#define CLogo_Process(p)   \
	Ccall(p, s_origCLogo_OnProcess, int, ())()
#define CBattle_Process(p) \
	Ccall(p, s_origCBattle_OnProcess, int, ())()
#define CTitle_Process(p) \
	Ccall(p, s_origCTitle_OnProcess, int, ())()
#define CSelect_Process(p) \
	Ccall(p, s_origCSelect_OnProcess, int, ())()

static DWORD s_origCLogo_OnProcess;
static DWORD s_origCBattle_OnProcess;
static DWORD s_origCTitle_OnProcess;
static DWORD s_origCSelect_OnProcess;

static bool s_swrapt;
static bool s_autoShutdown;

void setup_socket_client();

int __fastcall CLogo_OnProcess(void *This)
{
	int ret = 2;
	for (int i=0; i<3; i++) {
		ret = CLogo_Process(This);
		if(ret == 2) {
			if (__argc == 2) {
				if(CInputManager_ReadReplay(g_inputMgr, __argv[1])) {
					setup_socket_client();
					s_swrapt = true;
					// 入力があったように見せかける。END
					*(BYTE*)((DWORD)g_inputMgrs + 0x74) = 0xFF;
					// リプモードにチェンジ
					SetBattleMode(3, 2);
					ret = 6;
				}
			} else {
				//*(BYTE*)((DWORD)g_inputMgrs + 0x74) = 0xFF;
			
			}
		}
	}
	
	return ret;
}

typedef void(__stdcall *pSleep)(DWORD);

struct CharInput
{
	int lr;
	int ud;
	int a;
	int b;
	int c;
	int d;
	int ch;
	int s;
};

void * p1_obj = NULL;
void * p2_obj = NULL;

void write_operation(int operation, int which) {
    int first_d = operation / 9;
    int next_d = (operation % 9) / 3;
    int last_d = operation % 3;

	CharInput _input;
    _input.lr = 0;
    _input.ud = 0;
    _input.a = 0;
    _input.b = 0;
    _input.c = 0;
    _input.d = 0;
	_input.ch = 0;
	_input.s = 0;
    if (next_d == 1)
        _input.lr = -1;
	else if (next_d == 2)
        _input.lr = 1;
    if (last_d == 1)
        _input.ud = -1;
	else if (last_d == 2)
        _input.ud = 1;
    if (first_d == 1)
        _input.a = 1;
	else if (first_d == 2)
        _input.b = 1;
	else if (first_d == 3)
        _input.c = 1;
    else if (first_d == 4)
        _input.d = 1;
	if (which == 0)
		*(CharInput*)((char*)p1_obj + 0x754) = _input;
	else if (which == 1)
		*(CharInput*)((char*)p2_obj + 0x754) = _input;
}

int encode_action(int which) {
	CharInput _input;
	if (which == 0)
		_input = *(CharInput*)((char*)p1_obj + 0x754);
	else
		_input = *(CharInput*)((char*)p2_obj + 0x754);
	int first_d = 0, next_d = 0, last_d = 0;
    if (_input.ud < 0)
        last_d = 1;
    if (_input.ud > 0)
        last_d = 2;
    if (_input.lr < 0)
        next_d = 1;
    if (_input.lr > 0)
        next_d = 2;
    if (_input.a > 0)
        first_d = 1;
    if (_input.b > 0)
        first_d = 2;
    if (_input.c > 0)
        first_d = 3;
    if (_input.d > 0)
        first_d = 4;
    /*if (_input.ch > 0)
        now |= 256;
    if (_input.s > 0)
        now |= 512;*/
	return first_d * 9 + next_d * 3 + last_d;
}

SOCKET sclient;

void setup_socket_client()
{
	WORD sockVersion = MAKEWORD(2, 2);
	WSADATA data;
	int startup = 0;
	if((startup = WSAStartup(sockVersion, &data)) != 0)
	{
		FILE * fp = fopen("D:/1.log", "a+");
		fprintf(fp, "Connection Failed: Startup\n");
		if (startup == WSASYSNOTREADY)
			fprintf(fp, "System Not Ready\n");
		if (startup == WSAVERNOTSUPPORTED)
			fprintf(fp, "Version Not Supported\n");
		if (startup == WSAEINVAL)
			fprintf(fp, "DLL Invalid\n");
		fclose(fp);
		return;
	}

	sclient = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(sclient == INVALID_SOCKET)
	{
		FILE * fp = fopen("D:/1.log", "a+");
		fprintf(fp, "Connection Failed: Invalid Socket\n");
		fclose(fp);
		return;
	}

	sockaddr_in serAddr;
	serAddr.sin_family = AF_INET;
	serAddr.sin_port = htons(5211);
	serAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	if (connect(sclient, (sockaddr *)&serAddr, sizeof(serAddr)) == SOCKET_ERROR)
	{
		FILE * fp = fopen("D:/1.log", "a+");
		fprintf(fp, "Connection Failed: Connect Socket Error\n");
		fclose(fp);
		closesocket(sclient);
		return;
	}
}

static int z_ofs = 64, a_ofs = 76;
int z_time = 0, a_count = 0;

int one_update(void * This) {
	char recData[127] = "";
	int ret = recv(sclient, recData, 127, 0);
	if (ret > 0)
		recData[ret] = 0x00;

	void * root = *((void **)0x008855C4);
	p1_obj = *(void**)((char *)root + 0x0c);
	p2_obj = *(void**)((char *)root + 0x10);
	int p1_act, p2_act;
	sscanf(recData, "%d%d", &p1_act, &p2_act);
	write_operation(p1_act, 0);
	write_operation(p2_act, 1);

	// *(BYTE*)((DWORD)g_inputMgrs + z_ofs) = 1;
	int return_code = CBattle_Process(This);

	char sendData[255] = "";
	// Pos1X Pos1Y Pos2X Pos2Y Char1 Char2 Input1 Input2 Hp1 Hp2 WeatherID WeatherCounter
	sprintf(sendData, "%f %f %f %f %d %d %d %d %d %d %d %d",
				*(float *)((char *)p1_obj + 0xEC),
				*(float *)((char *)p1_obj + 0xF0),
				*(float *)((char *)p2_obj + 0xEC),
				*(float *)((char *)p2_obj + 0xF0),
				*(int *)ADDR_LCHARID,
				*(int *)ADDR_RCHARID,
				encode_action(0),
				encode_action(1),
				(int)*(short *)((char *)p1_obj + 0x184),
				(int)*(short *)((char *)p2_obj + 0x184),
				*(int *)(0x008841A0),
				*(int *)(0x008841AC));
	send(sclient, sendData, strlen(sendData), 0);
	/*FILE * fp = fopen("D:/1.log", "a+");
	fprintf(fp, "%d %d\n", ret, send(sclient, sendData, strlen(sendData), 0));
	fclose(fp);*/

	if (return_code != 5) {
		return_code = -1;
	}

	return return_code;
}

int __fastcall CTitle_OnProcess(void *This)
{
	//SetBattleMode(3, 1);
	//ret = 3;
	/*(*((DWORD *)This + 391)) = 3;
	*(BYTE*)((DWORD)g_inputMgrs + 0x74) = 0xff;
	if (z_time < 2)
		*(BYTE*)((DWORD)g_inputMgrs + z_ofs) = ++z_time;
	else
		*(BYTE*)((DWORD)g_inputMgrs + z_ofs) = z_time = 0;*/
	int ret = 2;
	for (int i=0; i<50; i++)
		ret = CTitle_Process(This);
	/*FILE * fp = fopen("D:/1.log", "a+");
	for (int i=0; i<=0x78; i++) {
		fprintf(fp, "%d\t", (int)*(BYTE*)((DWORD)g_inputMgrs + i));
	}
	fprintf(fp, "\n");
	fclose(fp);*/
	return ret;
}

int tmp_buffer[1000] = {-1};

int __fastcall CSelect_OnProcess(void *This)
{
	/*if (a_count < 3) {
		if (z_time < 2)
			*((DWORD *)This + 25) = *(BYTE*)((DWORD)g_inputMgrs + a_ofs) = ++z_time;
		else
			*((DWORD *)This + 25) = *(BYTE*)((DWORD)g_inputMgrs + a_ofs) = z_time = 0, a_count++;
	}
	else if (z_time < 2)
		*((DWORD *)This + 22) = *(BYTE*)((DWORD)g_inputMgrs + z_ofs) = ++z_time;
	else
		*((DWORD *)This + 22) = *(BYTE*)((DWORD)g_inputMgrs + z_ofs) = z_time = 0;*/
	*(int *)ADDR_LCHARID = rand() % 14;
	*(int *)ADDR_RCHARID = rand() % 14;
	*((DWORD *)This + 2) = 1;
	*((DWORD *)This + 4) = 0;
	*((DWORD *)This + 5) = 8935048;
	*((DWORD *)This + 35) = 255;
	*((DWORD *)This + 76) = 0;
	*((DWORD *)This + 81) = 8935104;
	*((DWORD *)This + 86) = 0;
	*((DWORD *)This + 91) = 0;
	*((DWORD *)This + 96) = 8935104;
	*((DWORD *)This + 101) = 8935108;
	*((DWORD *)This + 22) = *(BYTE*)((DWORD)g_inputMgrs + z_ofs) = 1;
	CSelect_Process(This);
	int ret = CSelect_Process(This);

	if (ret == 3) {
		// ret = 6;
	}
	//FILE * fp = fopen("D:/1.log", "a+");
	/*if (tmp_buffer[0] == -1) {
		for (int i=0; i<=0x200; i++)
			tmp_buffer[i] = (int)*((DWORD *)This + i);
	}
	for (int i=0; i<=0x200; i++) {
		if ((int)*((DWORD *)This + i) != tmp_buffer[i])
			fprintf(fp, "%d:%d\t", i, (int)*((DWORD *)This + i));
	}*/
	/*int * pt = *(int **)((char *)This + 9268);
	fprintf(fp, "%d\n", *((DWORD *)This + 16));
	// fprintf(fp, "\n");
	fclose(fp);*/
	return ret;
}

int __fastcall CBattle_OnProcess(void *This)
{
	int ret;
	static int rate = 500;
	for (int i=0; i<rate; i++) {
		ret = one_update(This);
		if (ret != 5) {
			return -1;
		}
	}
	return ret;
}

// 設定ロード
void LoadSettings(LPCSTR profilePath)
{
	// 自動シャットダウン
	s_autoShutdown = GetPrivateProfileInt("ReplayDnD", "AutoShutdown", 1, profilePath) != 0;
}

extern "C"
__declspec(dllexport) bool CheckVersion(const BYTE hash[16])
{
	return 1;
}

extern "C"
__declspec(dllexport) bool Initialize(HMODULE hMyModule, HMODULE hParentModule)
{
	char profilePath[1024 + MAX_PATH];

	GetModuleFileName(hMyModule, profilePath, 1024);
	PathRemoveFileSpec(profilePath);
	PathAppend(profilePath, "ReplayDnD.ini");
	LoadSettings(profilePath);

	DWORD old;
	::VirtualProtect((PVOID)rdata_Offset, rdata_Size, PAGE_EXECUTE_WRITECOPY, &old);
	s_origCLogo_OnProcess   = TamperDword(vtbl_CLogo   + 4, (DWORD)CLogo_OnProcess);
	s_origCBattle_OnProcess = TamperDword(vtbl_CBattle + 4, (DWORD)CBattle_OnProcess);
	// s_origCTitle_OnProcess  = TamperDword(vtbl_CTitle  + 4, (DWORD)CTitle_OnProcess);
	// s_origCSelect_OnProcess = TamperDword(vtbl_CSelect + 4, (DWORD)CSelect_OnProcess);
	::VirtualProtect((PVOID)rdata_Offset, rdata_Size, old, &old);

	::FlushInstructionCache(GetCurrentProcess(), NULL, 0);
	return true;
}

extern "C"
int APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpReserved)
{
	return TRUE;
}

