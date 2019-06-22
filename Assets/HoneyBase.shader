//Based on https://www.shadertoy.com/view/ttl3R2
Shader "Unlit/HoneyBase"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		iChannel1("iChannel1", CUBE) = "white" {}
		iChannel2("iChannel2", 3D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
			#pragma target 3.5
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

			samplerCUBE iChannel1;
			//sampler3D iChannel2;
			Texture3D iChannel2;
			float4 iChannel2_TexelSize;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }


			float3x3 rotX(float a)
			{
				return float3x3(1., 0., 0.,
					0., cos(a), sin(a),
					0., -sin(a), cos(a));
			}

			float3x3 rotY(float a)
			{
				return float3x3(cos(a), 0., sin(a),
					0., 1., 0.,
					-sin(a), 0., cos(a));
			}

			float3x3 rotZ(float a)
			{
				return float3x3(cos(a), sin(a), 0.,
					-sin(a), cos(a), 0.,
					0., 0., 1.);
			}

			// Smooth 3D texture interpolation
			float4 smoothSample(Texture3D tex, float3 p, int level)
			{
				//vec3 sz = vec3(textureSize(tex, 0));
				float3 sz = float3(32, 32, 32);

				int3 ip = int3(floor(p * sz));

				float4 position0 = float4((ip + int3(0, 0, 0)) & int3(sz - 1.), level);
				float4 position1 = float4((ip + int3(1, 0, 0)) & int3(sz - 1.), level);
				float4 position2 = float4((ip + int3(0, 1, 0)) & int3(sz - 1.), level);
				float4 position3 = float4((ip + int3(1, 1, 0)) & int3(sz - 1.), level);
				float4 position4 = float4((ip + int3(0, 0, 1)) & int3(sz - 1.), level);
				float4 position5 = float4((ip + int3(1, 0, 1)) & int3(sz - 1.), level);
				float4 position6 = float4((ip + int3(0, 1, 1)) & int3(sz - 1.), level);
				float4 position7 = float4((ip + int3(1, 1, 1)) & int3(sz - 1.), level);

				float4 s0 = tex.Load(position0);
				float4 s1 = tex.Load(position1);
				float4 s2 = tex.Load(position2);
				float4 s3 = tex.Load(position3);
				float4 s4 = tex.Load(position4);
				float4 s5 = tex.Load(position5);
				float4 s6 = tex.Load(position6);
				float4 s7 = tex.Load(position7);

				float3 f = smoothstep(0., 1., frac(p * sz));
				
				//return s0;
				return lerp(
					lerp(lerp(s0, s1, f.x),
						lerp(s2, s3, f.x), f.y),
					lerp(lerp(s4, s5, f.x),
						lerp(s6, s7, f.x), f.y),
					f.z) * abs(cos(_Time.y * -0.25));
			}

			static const float pi = acos(-1.);

			// Signed distance field
			float map(float3 p)
			{
				float d = 0.;

				p.x += 11.5;

				p += (smoothSample(iChannel2, p * 2., 0).rgb - .5) * .001;
				p += (smoothSample(iChannel2, p / 18. - 10. / 205. * float3(0, 0, 1), 0).rgb - .5) * 1.5;
				d += length(p.yz) - .6;

				d = max(d - .2, -(length(fmod(p, .2) - .1) - .015));

				d = min(d, length(abs(p.yz) - 1.5) - .05);
				d = min(d, length(p.yz - 1.2) - .05);

				return d;
			}

			float3 mapNormal(float3 p)
			{
				float3 e = float3(1e-3, 0., 0.);
				return normalize(float3(map(p + e.xyy) - map(p - e.xyy),
					map(p + e.yxy) - map(p - e.yxy),
					map(p + e.yyx) - map(p - e.yyx)));
			}

			float4 render(float2 fragCoord)
			{
				// Set up primary ray

				float2 p = fragCoord / _ScreenParams.xy * 2. - 1.;
				p.x *= _ScreenParams.x / _ScreenParams.y;

				float3 ro = float3(0, 0., 4.5);
				float3 rd = normalize(float3(p, -1.8));

				// Camera rotation

				//float3x3 m = rotY(.6) * rotZ(-.3);
				float3x3 m = mul(rotZ(-.33), rotY(.6));

				//rd = m * rd;
				rd = mul(rd, m);

				float4 fragColor = float4(0,0,0,0);

				// When flip is positive, the march is outside of the glass.
				// When it's negative, the march is inside of the glass.
				float flip = +1.;

				float3 transfer = float3(1, 1, 1);

				float3 rp = ro + rd * 2., prevhitrp = rp;

				for (int i = 0; i < 400; ++i)
				{
					float d = map(rp) * flip;

					// Test for surface hit
					if (abs(d) < 1e-4)
					{
						// Get the surface normal here
						float3 n = mapNormal(rp), on = n;

						// Put the normal and ray direction on the same side of the plane
						n *= -sign(dot(rd, n));

						// Fresnel term
						float fr = lerp(.01, .4, pow(clamp(1. - dot(-rd, n), 0., 1.), 5.));

						// Refract ray direction, or reflect if there is no solution (as per Snell's law).
						// This accounts for total internal reflection.
						float ior = 1.25;
						float3 refr = normalize(refract(normalize(rd), normalize(n), flip < 0. ? ior : 1. / ior));
						rd = dot(refr, refr) > 0. ? refr : reflect(rd, n);

						float dist = distance(rp, prevhitrp);

						// If the ray is just leaving a solid volume then aborb some energy
						// according to Beer's law.
						if (flip < 0.)
							//transfer *= exp(-abs(dist) * float3(.3, .5, .7) * 2.2);
							transfer *= exp(-abs(dist) * float3(.0122 * p.x, 0.003, .0003) * 13.2);

						// Just directly add a reflection here, to avoid the need for a branch path.
						// This isn't correct, but a reflection is needed somehow to get any kind of
						// convincing material appearance.
						//TODO fragColor.rgb += textureLod(iChannel1, reflect(rd, n), 2.).rgb * transfer * fr;
						float3 reflectVal = reflect(rd, n).xyz;
						float4 coordinate = float4(reflectVal.x, reflectVal.y, reflectVal.z, 2.);
						//fragColor.rgb += tex3Dlod(iChannel1, reflect(rd, n), 2.).rgb; *transfer * fr;
						fragColor.rgb += texCUBElod(iChannel1, coordinate).rgb *transfer * fr;


						prevhitrp = rp;

						flip = -flip;
						d = 2e-4;

						// Push the ray position through the surface along the normal.
						// This is more robust than pushing it along the ray's direction.
						rp += -n * 1e-3;

						transfer *= (1. - fr);
					}

					rp += rd * d * .3;

					// Test for far plane escape
					if (distance(rp, ro) > 15.)
						break;
				}

				float3 refc = float3(0, 0, 0);

				float wsum = 0.;

				// Filtered environment map lookup
				for (int z = -2; z < 2; ++z)
					for (int y = -2; y < 2; ++y)
						for (int x = -2; x < 2; ++x)
						{
							float w = 1. - float(max(abs(x), max(abs(y), abs(z)))) / 3.;
							//refc.rgb += textureLod(iChannel1, rd + float3(x, y, z) * .1, 3.).rgb * w;
							float4 position = float4(rd + float3(x, y, z) * .1, 3.);
							refc.rgb += texCUBElod(iChannel1, position).rgb * w;
							wsum += w;
						}

				fragColor.rgb += refc * transfer / wsum;

				// Vignet
				fragColor.rgb *= 1. - (pow(abs(p.x) / 2.2, 4.) + pow(abs(p.y) / 1.4, 4.)) * .7;

				fragColor.a = 1.;

				return fragColor;
			}

			// Halton sequence (radical inverse)
			float halton(const uint b, uint j)
			{
				float h = 0.0, f = 1.0 / float(b), fct = f;

				while (j > 0U)
				{
					h += float(j % b) * fct;
					j /= b;
					fct *= f;
				}

				return h;
			}


			// Sample unit disc
			float2 disc(float2 uv)
			{
				float a = uv.x * pi * 2.;
				float r = sqrt(uv.y);
				return float2(cos(a), sin(a)) * r;
			}

			float2 cone(float2 v)
			{
				return disc(float2(v.x, 1. - sqrt(1. - v.y)));
			}

			sampler2D iChannel3;

            fixed4 frag(v2f i) : SV_Target
            {

				float4 fragColor = float4(0, 0, 0, 0);

				//float4 oldColor = texelFetch(iChannel3, int2(fragCoord), 0);
				float4 position = float4(int2(i.uv.xy * _ScreenParams.xy), 0, 0);
				float4 oldColor = tex2Dlod(iChannel3, position) * 0.3;
				/*
				if (iMouse.z > .5)
					oldColor = float4(0);
				*/

				float2 uv = float2(halton(2U, uint(oldColor.w) & 2047U), halton(3U, uint(oldColor.w) & 2047U));

				float2 aaOffset = cone(uv) * 1.2;

				//fragColor = oldColor + float4(clamp(render(fragCoord + aaOffset).rgb, 0., 1.), 1.);
				fragColor = oldColor + float4(clamp(render((i.uv.xy * _ScreenParams.xy) + aaOffset).rgb, 0., 1.), 1.);
				//fragColor = oldColor + float4(clamp(render((i.uv.xy+ aaOffset)*_ScreenParams.xy).rgb, 0., 1.), 1.);
				return fragColor;
            }
            ENDCG
        }
    }
}
