using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Noise3DTexGenerator : MonoBehaviour
{
    Texture3D texture;

    void Start()
    {
        texture = CreateTexture3D(32);
        // set 3D tex to material
        GetComponent<Renderer>().material.SetTexture("iChannel2", texture);
    }

    Texture3D CreateTexture3D(int size)
    {
        Color[] colorArray = new Color[size * size * size];
        texture = new Texture3D(size, size, size, TextureFormat.RGBA32, true);
        float r = 1.0f / (size - 1.0f);
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                for (int z = 0; z < size; z++)
                {
                    //Color c = new Color(x * r, y * r, z * r, 1.0f);
                    Color c = new Color(Random.value, Random.value, Random.value, 1.0f);
                    //Color c = Color.white;
                    colorArray[x + (y * size) + (z * size * size)] = c;
                }
            }
        }
        texture.SetPixels(colorArray);
        texture.Apply();
        return texture;
    }

    // Update is called once per frame
    void Update()
    {
        //texture = CreateTexture3D(32);
        //GetComponent<Renderer>().material.SetTexture("iChannel2", texture);
    }
}
