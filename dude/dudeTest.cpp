
#ifdef _XBOX
#include <xtl.h>
#include <xboxmath.h>
#include <tracerecording.h>

#define PIX_TRACE

#else
#include <windows.h>
#include <xmmintrin.h>
#endif

#include <stdio.h>
#include <vector>
#include <math.h>




struct Vec2
{
	inline Vec2() {}
	inline Vec2(float _x, float _y)
		: x(_x), y(_y) {}
	float x;
	float y;

	inline Vec2 operator + (const Vec2& v)	const {	return Vec2( x+v.x, y+v.y );	}
	inline Vec2 operator - (const Vec2& v) const {	return Vec2( x-v.x, y-v.y );	}

	inline Vec2& operator += (const Vec2& v) { x += v.x; y += v.y; return *this; }
	inline Vec2& operator -= (const Vec2& v)  { x -= v.x; y -= v.y; return *this; }
};

namespace Vec
{
	void Normalize(Vec2& v);
	
	inline Vec2 Scale( const Vec2& a, float scale )
	{
		return Vec2( a.x * scale, a.y * scale );
	}

	inline float LengthSq( const Vec2& v )
	{
		return v.x*v.x + v.y*v.y;
	}

	inline float Length( const Vec2& v )
	{
		return sqrt( LengthSq( v ) );
	}

	inline Vec2 GetNormal(const Vec2& v)
	{
		float len = Length( v );
		return Scale( v, 1.0f/len );
	}

	inline float Dot( const Vec2& v, const Vec2& w)
	{
		return v.x*w.x + v.y*w.y;
	}
}


struct RandomSet
{
	float GetNormalisedFloat();	// returns number between 0 and 1
	int GetRangedInt(int from, int to);	// returns number from >= rand() <= to
};

float RandomSet::GetNormalisedFloat()
{
	return (float)rand() / (float)RAND_MAX; 
}

int RandomSet::GetRangedInt(int from, int to)
{
	int len = to - from;
	return from + (int)(GetNormalisedFloat() * len);
}

float clamp(float min, float val, float max)
{
	if( val < min )
		return min;
	if( val > max )
		return max;
	return val;
}

template <typename T>
struct RandomPoolAllocator
{
public:
	void Construct( int size, RandomSet& rand )
	{
		pool = new T[size*10];
		randomIndices.clear();
		randomIndices.reserve( size );

		int i;
		for( i=0; i<size; i++ )
			randomIndices.push_back(i);
		for( int i=0; i<size; i++ )
			std::swap( randomIndices[i], randomIndices[ rand.GetRangedInt(0, size) ] );
	}

	void Destruct()
	{
		delete [] pool;
		randomIndices.clear();
	}

	T* allocate()
	{
		int i = randomIndices.back();
		randomIndices.pop_back();
		return &pool[i*10];
	}

private:
	T* pool;
	std::vector<int> randomIndices;
};


/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

struct Timer
{
	void Start()
	{
		QueryPerformanceFrequency( &freq );
		QueryPerformanceCounter( &start );
	}
	double End()
	{
		LARGE_INTEGER end;
		QueryPerformanceCounter( &end );

		double elapsed = (double) end.QuadPart - start.QuadPart;
		elapsed /= (double) freq.QuadPart;
		return elapsed;
	}

	LARGE_INTEGER start, freq;	
};


/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

float neighbourRange = 10.0f;


struct World
{
	Vec2 extents;
	Vec2 halfExtents;
};


struct Dude_Original
{
	RandomSet rand;
	Vec2 position;
	char someData[192];
	Vec2 targetPos;
	float targetPosValid;
	float speed;
	float heading;
	float age;

	void Construct( const RandomSet& _rand, const World& world )
	{
		position.x = (rand.GetNormalisedFloat() * world.extents.x) - world.halfExtents.x;
		position.y = (rand.GetNormalisedFloat() * world.extents.y) - world.halfExtents.y;
		heading = rand.GetNormalisedFloat() * 2.0f * 3.142f;
		age = 0.0f;
		speed = rand.GetNormalisedFloat() + 5.0f;
		targetPos = Vec2(0.0f, 0.0f);
	}

	void SearchNeighbours(Dude_Original** dudes, int numDudes, int thisDude)
	{
		float neighbourRangeSq = neighbourRange*neighbourRange;

		Vec2 avg_pos(-dudes[thisDude]->position.x, -dudes[thisDude]->position.y);
		float numNeighbours = 1.0f;

		// search for nearby dudes and head for the center
		for(int i=0; i<numDudes; ++i)
		{
			//if(i != thisDude)
			{
				Vec2 diff = dudes[i]->position - position;
				if( Vec::LengthSq(diff) < neighbourRangeSq )
				{
					avg_pos += dudes[i]->position;
					numNeighbours += 1.0f;
				}
			}
		}

		if( numNeighbours != 1.0f )
		{
			targetPosValid = 1.0f;
			targetPos = Vec::Scale( avg_pos, 1.0f/numNeighbours );
		}
		else
		{
			targetPosValid = 0.0f;
		}
	}

	void Update(float dt)
	{
		float maxTurnPerSecond = 1.0f;
		float maxTurn = maxTurnPerSecond * dt;

		{
			// turn towards targetPos
			Vec2 dir = Vec::GetNormal( targetPos - position );
			Vec2 currentHeading( sin(heading), cos(heading) );
			float angleDiff = Vec::Dot( dir, currentHeading );
			angleDiff = clamp( -maxTurn, angleDiff, maxTurn );
			heading += angleDiff * targetPosValid;
		}
		
		float targetPosInvalid = 1.0f - targetPosValid;
		{
			// turn slightly
			float turnAmount = 0.1f * dt;
			heading += targetPosInvalid * (( rand.GetNormalisedFloat() / turnAmount ) - (turnAmount/2.0f));
		}

		Vec2 newHeading( sin(heading), cos(heading) );
		position = position + Vec::Scale(newHeading, speed); 
	}

};

//////////////////////////////////////////////////
//////////////////////////////////////////////////

World world = { Vec2(100.0f, 100.0f), Vec2(50.0f, 50.0f) };
#ifdef _DEBUG
int numDudes = 1000;
#else
int numDudes = 1000;
#endif

#ifdef PIX_TRACE
int numIterations = 2;
#else
int numIterations = 100;
#endif
float frameTime = 0.066f;

//////////////////////////////////////////////////
//////////////////////////////////////////////////


void TimeOnePass_Original(double& search, double& update, double& searchContig, double& updateContig)
{
	int i,j;

	{
		RandomPoolAllocator<Dude_Original> dudePool;
		{
			RandomSet rand;
			dudePool.Construct( numDudes, rand );
		}

		std::vector<Dude_Original*> dudes;
		dudes.reserve( numDudes );

		//std::vector<Dude_Original> dudesContig;
		Dude_Original* dudesContig = new Dude_Original [ numDudes ];
		std::vector<Dude_Original*> dudesContigPtrs;
		//dudesContig.resize( numDudes );
		dudesContigPtrs.resize( numDudes );

		for( i=0; i<numDudes; ++i )
		{
			RandomSet rand;
			Dude_Original* dude = dudePool.allocate();
			dude->Construct( rand, world );
			dudes.push_back( dude );

			dudesContig[i].Construct( rand, world );
			dudesContigPtrs[i] = &dudesContig[i];
		}

#if defined(_XBOX) && defined(PIX_TRACE)
		XTraceStartRecording( "e:\\dude_original.pix2" );
#endif

		double step1 = 0.0, step2 = 0.0, step1contig = 0.0, step2contig = 0.0;

		for( i=0; i<numIterations; i++ )
		{
			for( j=0; j<numDudes; j++ )
			{
				Dude_Original* thisDude = dudes[j];
				Dude_Original* thisDudeContig = &dudesContig[j];

				Timer subTimer;
				subTimer.Start();
				thisDude->SearchNeighbours( &dudes[0], dudes.size(), j );
				step1 += subTimer.End() * 10000000.0;

				subTimer.Start();
				thisDude->Update( frameTime );
				step2 += subTimer.End() * 10000000.0;

#if !defined(PIX_TRACE)
// don't confuse the trace
				subTimer.Start();
				thisDudeContig->SearchNeighbours( &dudesContigPtrs[0], dudesContigPtrs.size(), j );
				step1contig += subTimer.End() * 10000000.0;

				subTimer.Start();
				thisDudeContig->Update( frameTime );
				step2contig += subTimer.End() * 10000000.0;
#endif
			}
		}

#if defined(_XBOX) && defined(PIX_TRACE)
		XTraceStopRecording();
#endif

		step1 /= (numIterations * numDudes);
		step2 /= (numIterations * numDudes);
		step1contig /= (numIterations * numDudes);
		step2contig /= (numIterations * numDudes);

		search += step1;
		update += step2;
		searchContig += step1contig;
		updateContig += step2contig;

		printf( "Dude_Original:  %2.2f %2.2f %2.2f %2.2f\n", step1, step2, step1contig, step2contig );

		delete [] dudesContig;
		dudePool.Destruct();

		//return timer.End();
	}
}



// OK now let's try the DoD method


struct Dude_Stream
{
	RandomSet* __restrict rand;
	Vec2*__restrict  position;
	float* __restrict speed;
	float* __restrict heading;
	float* __restrict age;
	float* __restrict targetPosValid;
	Vec2* __restrict targetPos;

	int count;

	void Construct( RandomSet& _rand, const World& world, int _count )
	{
		count = _count;
		rand = new RandomSet[ _count ];
		position = (Vec2*)_aligned_malloc( sizeof(Vec2)*_count, 16 );
		speed = new float[ _count ];
		heading = new float[ _count ];
		age = new float[ _count ];
		targetPosValid = new float[ _count ];
		targetPos = new Vec2[ _count ];

		for( int i=0; i<_count; i++ )
		{
			position[i].x = (_rand.GetNormalisedFloat() * world.extents.x) - world.halfExtents.x;
			position[i].y = (_rand.GetNormalisedFloat() * world.extents.y) - world.halfExtents.y;
			heading[i] = _rand.GetNormalisedFloat() * 2.0f * 3.142f;
			age[i] = 0.0f;
			speed[i] = 0.5f + _rand.GetNormalisedFloat();
			targetPosValid[i] = 0.0f;
			targetPos[i] = Vec2(0.0f, 0.0f);
		}
	}

	~Dude_Stream()
	{
		delete [] rand;
		_aligned_free( position );
		delete [] speed;
		delete [] heading; 
		delete [] age;
		delete [] targetPosValid;
		delete [] targetPos;
	}
	



	static void SearchNeighbours(Dude_Stream& dudes)
	{
		float neighbourRangeSq = neighbourRange*neighbourRange;

		// search for nearby dudes and head for the center
		for(int i=0; i<dudes.count; ++i)
		{
			const Vec2 pos = dudes.position[i];
			Vec2 avg_pos(-pos.x, -pos.y);
			float numNeighbours = 1.0f;

			for(int j=0; j<dudes.count; ++j)
			{
				Vec2 diff = dudes.position[j] - pos;
				float distSq = Vec::LengthSq(diff);

				if( distSq < neighbourRangeSq )
				{
					avg_pos += dudes.position[j];
					numNeighbours += 1.0f;
				}
			}

			if( numNeighbours!=1.0f )
			{
				dudes.targetPosValid[i] = 1.0f;
				dudes.targetPos[i] = Vec::Scale( avg_pos, 1.0f/numNeighbours );
			}
			else
				dudes.targetPosValid[i] = 0.0f;
		}

	}


	static void Update(float dt, Dude_Stream& dudes, RandomSet& rand)
	{
		for( int i=0; i<dudes.count; i++ )
		{
			float maxTurnPerSecond = 1.0f;
			float maxTurn = maxTurnPerSecond * dt;

			float heading = dudes.heading[i];

			{
				// turn towards targetPos
				Vec2 dir = Vec::GetNormal( dudes.targetPos[i] - dudes.position[i]);
				Vec2 currentHeading( sin(heading), cos(heading) );
				float angleDiff = Vec::Dot( dir, currentHeading );
				angleDiff = clamp( -maxTurn, angleDiff, maxTurn );
				heading += angleDiff * dudes.targetPosValid[i];
			}
			
			float targetPosInvalid = 1.0f - dudes.targetPosValid[i];
			{
				// turn slightly
				float turnAmount = 0.1f * dt;
				heading += targetPosInvalid * (( rand.GetNormalisedFloat() / turnAmount ) - (turnAmount/2.0f));
			}

			dudes.heading[i] = heading;

			Vec2 newHeading( sin(heading), cos(heading) );
			dudes.position[i] = dudes.position[i] + Vec::Scale(newHeading, dudes.speed[i]); 
		}
	}



	static void SearchNeighboursSIMD( Dude_Stream& dudes )
	{
#ifndef _XBOX
		__m128 neighbourRangeSq = _mm_set1_ps( neighbourRange*neighbourRange );

		// search for nearby dudes and head for the center
		// search 4 dudes at a time
		for(int i=0; i<dudes.count; ++i)
		{
			__m128 avg_posX = _mm_set_ss( -dudes.position[i].x );
			__m128 avg_posY = _mm_set_ss( -dudes.position[i].y );

			__m128 numNeighbours = _mm_set_ss( -1.0f );
			__m128 allNeighbours = _mm_set1_ps( 1.0f );
			
			__m128 posX = _mm_set1_ps( dudes.position[i].x );
			__m128 posY = _mm_set1_ps( dudes.position[i].y );

			for(int j=0; j<dudes.count; j += 4)
			{
				//if(i != j)
				{
					// we want a register full of x values and a register full of y values
					// loading SSE registers is slightly tricky because of the use of Vec2[x,y] pair
					// if was stored as 2 float streams it would be easy!

					// input: x1y1x2y3, x3y3x4y4
					// output [x1x2x3x4] [y1y2y3y4]
					
					__m128 a = _mm_load_ps( &dudes.position[j].x );
					__m128 b = _mm_load_ps( &dudes.position[j+2].x );

					__m128 dudesX = _mm_shuffle_ps( a, b, _MM_SHUFFLE(2,0,2,0) );
					__m128 dudesY = _mm_shuffle_ps( a, b, _MM_SHUFFLE(3,1,3,1) );


					//__m128 dudesX = _mm_set_ps( dudes.position[j+0].x, dudes.position[j+1].x, dudes.position[j+2].x, dudes.position[j+3].x );
					//__m128 dudesX = _mm_set_ps( dudes.position[j+0].y, dudes.position[j+1].y, dudes.position[j+2].y, dudes.position[j+3].y );

					__m128 diffX = _mm_sub_ps( posX, dudesX );
					__m128 diffY = _mm_sub_ps( posY, dudesY );

					// compute distSq (x*x + y*y)
					__m128 distSq = _mm_add_ps( _mm_mul_ps( diffX, diffX ), _mm_mul_ps( diffY, diffY ) );

					// compare with neighbourRangeSq
					__m128 result = _mm_cmplt_ps( distSq, neighbourRangeSq );


					// avg_posX = avg_posX + (result & dudesX)
					// avg_posY = avg_posY + (result & dudesY)
					// numNeighbours = numNeighbours + (result & (1,1,1,1))

					avg_posX = _mm_add_ps( avg_posX, _mm_and_ps( result, dudesX ) );
					avg_posY = _mm_add_ps( avg_posY, _mm_and_ps( result, dudesY ) );

					numNeighbours = _mm_add_ps( numNeighbours, _mm_and_ps( result, allNeighbours ) );
				}
			}

			// sum the num neighbours
			float fNeighbours = numNeighbours.m128_f32[0] + numNeighbours.m128_f32[1] + numNeighbours.m128_f32[2] + numNeighbours.m128_f32[3];

			if( fNeighbours > 1.0f )
			{
				__m128 totalNeighbours = _mm_set1_ps( fNeighbours );

				// average the positions
				avg_posX = _mm_div_ps( avg_posX, totalNeighbours );
				avg_posY = _mm_div_ps( avg_posY, totalNeighbours );

				// combine into a single value
				dudes.targetPos[i].x = ( avg_posX.m128_f32[0] + avg_posX.m128_f32[1] + avg_posX.m128_f32[2] + avg_posX.m128_f32[3] );
				dudes.targetPos[i].y = ( avg_posY.m128_f32[0] + avg_posY.m128_f32[1] + avg_posY.m128_f32[2] + avg_posY.m128_f32[3] );

				dudes.targetPosValid[i] = 1.0f;
			}
			else
				dudes.targetPosValid[i] = 0.0f;
		}
#endif
	}

};


void TimeOnePass_Stream(double& search, double& simdSearch, double& update)
{
	int i;

	{
		RandomSet rand;
		Dude_Stream dudes;
		dudes.Construct( rand, world, numDudes );

#if defined(_XBOX) && defined(PIX_TRACE)
		XTraceStartRecording( "e:\\dude_stream.pix2" );
#endif

		double step1 = 0.0, step2 = 0.0, step3 = 0.0;

		Timer timer;
		timer.Start();

		for( i=0; i<numIterations; i++ )
		{
			Timer subTimer;
			subTimer.Start();
			Dude_Stream::SearchNeighbours( dudes );
			step1 += subTimer.End() * 10000000.0;

			subTimer.Start();
			Dude_Stream::SearchNeighboursSIMD( dudes );
			step2 += subTimer.End() * 10000000.0;

			subTimer.Start();
			Dude_Stream::Update( frameTime, dudes, rand );
			step3 += subTimer.End() * 10000000.0;
		}

#if defined(_XBOX) && defined(PIX_TRACE)
		XTraceStopRecording();
#endif

		step1 /= (numIterations * numDudes);
		step2 /= (numIterations * numDudes);
		step3 /= (numIterations * numDudes);

		search += step1;
		simdSearch += step2;
		update += step3;

		printf( "Dude_Stream:  %2.2f %2.2f %2.2f %2.2f\n", step1, step2, step3, timer.End() );
	}
}


void dude_main()
{
	printf("===============================================\n");
	printf("numDudes: %d      numIterations: %d\n\n", numDudes, numIterations);

#ifdef PIX_TRACE
	int numPasses = 1;
#else
	int numPasses = 3;
#endif
	int pass;
	double origSearch=0.0, origUpdate=0.0, contigSearch=0.0, contigUpdate=0.0;
	for( pass=0; pass < numPasses; pass++ )
	{
		TimeOnePass_Original(origSearch, origUpdate, contigSearch, contigUpdate);		
	}
	double streamSearch=0.0, simdSearch=0.0, streamUpdate=0.0;
	for( pass=0; pass < numPasses; pass++ )
	{
		TimeOnePass_Stream(streamSearch, simdSearch, streamUpdate);
	}

	origSearch /= numPasses;
	origUpdate /= numPasses;
	contigSearch /= numPasses;
	contigUpdate /= numPasses;
	streamSearch /= numPasses;
	simdSearch /= numPasses;
	streamUpdate /= numPasses;

	printf("\n===============================================\n");
	printf(" Average times for searching neighbours per dude:\n");
	printf("    Original: %2.2f\n", origSearch );
	printf("    Contig  : %2.2f (x %2.2f)\n", contigSearch, origSearch/contigSearch );
	printf("    Stream  : %2.2f (x %2.2f)\n", streamSearch, origSearch/streamSearch );
	printf("    SIMD    : %2.2f (x %2.2f)\n", simdSearch, origSearch/simdSearch );

	printf("\n Average times for updating dude:\n");
	printf("    Original: %2.2f\n", origUpdate );
	printf("    Contig  : %2.2f (x %2.2f)\n", contigUpdate, origUpdate/contigUpdate );
	printf("    Stream  : %2.2f (x %2.2f)\n", streamUpdate, origUpdate/streamUpdate );

}



int main(void**, int)
{
	dude_main();

	for(;;) {Sleep(10);}

	return 0;
}

