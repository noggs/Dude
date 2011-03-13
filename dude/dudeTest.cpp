
#ifdef XBOX
#include "stdafx.h"
#else
#include <windows.h>
#endif

#include <stdio.h>
#include <vector>
#include <math.h>

#include <xmmintrin.h>


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

template <typename T>
struct RandomPoolAllocator
{
public:
	void Construct( int size, RandomSet& rand )
	{
		pool = new T[size];
		randomIndices.reserve( size );

		int i;
		for( i=0; i<size; i++ )
			randomIndices.push_back(i);
		//for( int i=0; i<size; i++ )
		//	std::swap( randomIndices[i], randomIndices[ rand.GetRangedInt(0, size) ] );
	}

	T* allocate()
	{
		int i = randomIndices.back();
		randomIndices.pop_back();
		return &pool[i];
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
	bool targetPosValid;
	float speed;
	float heading;
	float age;

	void Construct( const RandomSet& _rand, const World& world )
	{
		position.x = (rand.GetNormalisedFloat() * world.extents.x) - world.halfExtents.x;
		position.y = (rand.GetNormalisedFloat() * world.extents.y) - world.halfExtents.y;
		heading = rand.GetNormalisedFloat() * 2.0f * 3.142f;
		age = 0.0f;
	}

	void SearchNeighbours(Dude_Original** dudes, int numDudes, int thisDude)
	{
		float neighbourRangeSq = neighbourRange*neighbourRange;

		Vec2 avg_pos(-dudes[thisDude]->position.x, -dudes[thisDude]->position.y);
		int numNeighbours = 1;

		// search for nearby dudes and head for the center
		for(int i=0; i<numDudes; ++i)
		{
			//if(i != thisDude)
			{
				Vec2 diff = dudes[i]->position - position;
				if( Vec::LengthSq(diff) < neighbourRangeSq )
				{
					avg_pos += dudes[i]->position;
					numNeighbours++;
				}
			}
		}

		targetPosValid = (numNeighbours!=0);
		targetPos = (numNeighbours!=0) ? Vec::Scale( avg_pos, 1.0f/(float)numNeighbours ) : Vec2();
	}

	void Update(float dt)
	{
		// turn towards targetPos
		//Vec2 dir = Vec::GetNormal( targetPos - position );

		//Vec2 currentHeading( sin(heading), cos(heading) );

		//float angleDiff = Vec::Dot( dir, currentHeading );
		

		// turn slightly
		float turnAmount = 0.1f * dt;
		heading += ( rand.GetNormalisedFloat() / turnAmount ) - (turnAmount/2.0f);
		Vec2 newHeading( sin(heading), cos(heading) );

		//newHeading = dir;

		position = position + Vec::Scale(newHeading, speed); 
	}

};

//////////////////////////////////////////////////
//////////////////////////////////////////////////

World world = { Vec2(100.0f, 100.0f), Vec2(50.0f, 50.0f) };
#ifdef _DEBUG
int numDudes = 100;
#else
int numDudes = 5000;
#endif
int numIterations = 60;
float frameTime = 0.066f;

//////////////////////////////////////////////////
//////////////////////////////////////////////////


void TimeOnePass_Original(double& search, double& update)
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

		for( i=0; i<numDudes; ++i )
		{
			RandomSet rand;
			Dude_Original* dude = dudePool.allocate();
			dude->Construct( rand, world );
			dudes.push_back( dude );
		}

		Timer timer;
		timer.Start();

		double step1 = 0.0, step2 = 0.0;

		for( i=0; i<numIterations; i++ )
		{
			for( j=0; j<numDudes; j++ )
			{
				Timer subTimer;
				subTimer.Start();

				dudes[j]->SearchNeighbours( &dudes[0], dudes.size(), j );

				step1 += subTimer.End() * 10000000.0;
				subTimer.Start();

				dudes[j]->Update( frameTime );

				step2 += subTimer.End() * 10000000.0;
			}
		}

		step1 /= (numIterations * numDudes);
		step2 /= (numIterations * numDudes);

		search += step1;
		update += step2;

		printf( "Dude_Original:  %2.2fus %2.2fus %2.2fs\n", step1, step2, timer.End() );

		//return timer.End();
	}
}



// OK now let's try the DoD method


struct Dude_Stream
{
	RandomSet* rand;
	Vec2* position;
	float* speed;
	float* heading;
	float* age;
	bool* targetPosValid;
	Vec2* targetPos;

	float* positionX;
	float* positionY;

	int count;

	void Construct( RandomSet& _rand, const World& world, int _count )
	{
		count = _count;
		rand = new RandomSet[ _count ];
		//position = new Vec2[ _count ];
		position = (Vec2*)_aligned_malloc( sizeof(Vec2)*_count, 16 );
		speed = new float[ _count ];
		heading = new float[ _count ];
		age = new float[ _count ];
		targetPosValid = new bool[ _count ];
		targetPos = new Vec2[ _count ];

		positionX = new float[ _count ];
		positionY = new float[ _count ];

		for( int i=0; i<_count; i++ )
		{
			positionX[i] = position[i].x = (_rand.GetNormalisedFloat() * world.extents.x) - world.halfExtents.x;
			positionY[i] = position[i].y = (_rand.GetNormalisedFloat() * world.extents.y) - world.halfExtents.y;
			heading[i] = _rand.GetNormalisedFloat() * 2.0f * 3.142f;
			age[i] = 0.0f;
			speed[i] = 0.5f + _rand.GetNormalisedFloat();
			targetPosValid[i] = false;
		}
	}

	~Dude_Stream()
	{
		delete [] rand;
		//delete [] position;
		_aligned_free( position );
		delete [] speed;
		delete [] heading; 
		delete [] age;
		delete [] targetPosValid;
		delete [] targetPos;

		delete [] positionX;
		delete [] positionY;
	}
	



	static void SearchNeighbours(Dude_Stream& dudes)
	{
		float neighbourRangeSq = neighbourRange*neighbourRange;

		// search for nearby dudes and head for the center
		for(int i=0; i<dudes.count; ++i)
		{
			const Vec2 pos = dudes.position[i];
			Vec2 avg_pos(-pos.x, -pos.y);
			int numNeighbours = 1;

			for(int j=0; j<dudes.count; ++j)
			{
				//const Vec2 otherPos = dudes.position[j];
				Vec2 diff = dudes.position[j] - pos;
				float distSq = Vec::LengthSq(diff);

				if( distSq < neighbourRangeSq )
				{
					avg_pos += dudes.position[j];
					numNeighbours++;
				}
			}

			if( numNeighbours!=1 )
			{
				dudes.targetPosValid[i] = true;
				dudes.targetPos[i] = Vec::Scale( avg_pos, 1.0f/(float)numNeighbours );
			}
			else
				dudes.targetPosValid[i] = false;
		}

	}


	static void Update(float dt, Dude_Stream& dudes, RandomSet& rand)
	{
		for( int i=0; i<dudes.count; i++ )
		{
			// turn slightly
			float turnAmount = 0.1f * dt;
			dudes.heading[i] += ( rand.GetNormalisedFloat() / turnAmount ) - (turnAmount/2.0f);
			Vec2 newHeading( sin(dudes.heading[i]), cos(dudes.heading[i]) );
			dudes.position[i] = dudes.position[i] + Vec::Scale(newHeading, dudes.speed[i]); 
		}
	}



	static void SearchNeighboursSIMD( Dude_Stream& dudes )
	{
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

				dudes.targetPosValid[i] = true;
			}
			else
				dudes.targetPosValid[i] = false;
		}
	}


	static void SearchNeighboursSIMD2( Dude_Stream& dudes )
	{
		__m128 neighbourRangeSq = _mm_set1_ps( neighbourRange*neighbourRange );

		// search for nearby dudes and head for the center
		// search 4 dudes at a time
		for(int i=0; i<dudes.count; ++i)
		{
			__m128 avg_posX = _mm_setzero_ps();
			__m128 avg_posY = _mm_setzero_ps();

			__m128 numNeighbours = _mm_setzero_ps();
			__m128 allNeighbours = _mm_set1_ps( 1.0f );

			__m128 posX = _mm_set1_ps( dudes.positionX[i] );
			__m128 posY = _mm_set1_ps( dudes.positionY[i] );

			for(int j=0; j<dudes.count; j += 4)
			{
				//if(i != j)
				{
					__m128 dudesX = _mm_load_ps( &dudes.position[j].x );
					__m128 dudesY = _mm_load_ps( &dudes.position[j+2].x );

					__m128 diffX = _mm_sub_ps( posX, dudesX );
					__m128 diffY = _mm_sub_ps( posY, dudesY );

					// compute distSq (x*x + y*y)
					__m128 distSq = _mm_add_ps( _mm_mul_ps( diffX, diffX ), _mm_mul_ps( diffY, diffY ) );

					// compare with neighbourRangeSq
					__m128 result = _mm_cmplt_ps( distSq, neighbourRangeSq );

					avg_posX = _mm_add_ps( avg_posX, _mm_and_ps( result, dudesX ) );
					avg_posY = _mm_add_ps( avg_posY, _mm_and_ps( result, dudesY ) );

					numNeighbours = _mm_add_ps( numNeighbours, _mm_and_ps( result, allNeighbours ) );
				}
			}

			// sum the num neighbours
			float fNeighbours = numNeighbours.m128_f32[0] + numNeighbours.m128_f32[1] + numNeighbours.m128_f32[2] + numNeighbours.m128_f32[3];

			if( fNeighbours > 0 )
			{
				__m128 totalNeighbours = _mm_set1_ps( fNeighbours );

				// average the positions
				avg_posX = _mm_div_ps( avg_posX, totalNeighbours );
				avg_posY = _mm_div_ps( avg_posY, totalNeighbours );

				// combine into a single value
				dudes.targetPos[i].x = ( avg_posX.m128_f32[0] + avg_posX.m128_f32[1] + avg_posX.m128_f32[2] + avg_posX.m128_f32[3] );
				dudes.targetPos[i].y = ( avg_posY.m128_f32[0] + avg_posY.m128_f32[1] + avg_posY.m128_f32[2] + avg_posY.m128_f32[3] );

				dudes.targetPosValid[i] = true;
			}
			else
				dudes.targetPosValid[i] = false;
		}
	}

};


void TimeOnePass_Stream(double& search, double& simdSearch, double& update)
{
	int i;

	{
		RandomSet rand;
		Dude_Stream dudes;
		dudes.Construct( rand, world, numDudes );

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
	int numPasses = 5;
	int pass;
	double origSearch=0.0, origUpdate=0.0;
	for( pass=0; pass < numPasses; pass++ )
	{
		TimeOnePass_Original(origSearch, origUpdate);		
	}
	double streamSearch=0.0, simdSearch=0.0, streamUpdate=0.0;
	for( pass=0; pass < numPasses; pass++ )
	{
		TimeOnePass_Stream(streamSearch, simdSearch, streamUpdate);
	}

	origSearch /= numPasses;
	origUpdate /= numPasses;
	streamSearch /= numPasses;
	simdSearch /= numPasses;
	streamUpdate /= numPasses;

	printf("\n===============================================\n");
	printf(" Average times for searching neighbours per dude:\n");
	printf("    Original: %2.2f\n", origSearch );
	printf("    Stream  : %2.2f (x %2.2f)\n", streamSearch, origSearch/streamSearch );
	printf("    SIMD    : %2.2f (x %2.2f)\n", simdSearch, origSearch/simdSearch );

	printf("\n Average times for updating dude:\n");
	printf("    Original: %2.2f\n", origUpdate );
	printf("    Stream  : %2.2f (x %2.2f)\n", streamUpdate, origUpdate/streamUpdate );

}




