#ifdef _XBOX
#include <xtl.h>
#include <xboxmath.h>
#else
#include <windows.h>
#endif

void dude_main();

int main(void**, int)
{
	dude_main();

	for(;;) {Sleep(10);}

	return 0;
}


