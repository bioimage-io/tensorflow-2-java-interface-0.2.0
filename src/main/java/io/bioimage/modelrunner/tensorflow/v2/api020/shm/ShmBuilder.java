/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.tensorflow.v2.api020.shm;

import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A utility class that converts {@link Tensor}s into {@link SharedMemoryArray}s for
 * interprocessing communication
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ShmBuilder
{
    /**
     * Utility class.
     */
    private ShmBuilder()
    {
    }

    /**
     * Create a {@link SharedMemoryArray} from a {@link Tensor}
     * @param tensor
     * 	the tensor to be passed into the other process through the shared memory
     * @param memoryName
     * 	the name of the memory region where the tensor is going to be copied
     * @throws IllegalArgumentException if the data type of the tensor is not supported
     * @throws IOException if there is any error creating the shared memory array
     */
    @SuppressWarnings("unchecked")
	public static void build(Tensor<? extends TType>  tensor, String memoryName) throws IllegalArgumentException, IOException
    {
		switch (tensor.dataType().name())
        {
            case TUint8.NAME:
            	buildFromTensorUByte((Tensor<TUint8>) tensor, memoryName);
            	break;
            case TInt32.NAME:
            	buildFromTensorInt((Tensor<TInt32>) tensor, memoryName);
            	break;
            case TFloat32.NAME:
            	buildFromTensorFloat((Tensor<TFloat32>) tensor, memoryName);
            	break;
            case TFloat64.NAME:
            	buildFromTensorDouble((Tensor<TFloat64>) tensor, memoryName);
            	break;
            case TInt64.NAME:
            	buildFromTensorLong((Tensor<TInt64>) tensor, memoryName);
            	break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

    private static void buildFromTensorUByte(Tensor<TUint8> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new UnsignedByteType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 1;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.rawData().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorInt(Tensor<TInt32> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 4;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.rawData().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorFloat(Tensor<TFloat32> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new FloatType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 4;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.rawData().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorDouble(Tensor<TFloat64> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new DoubleType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 8;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.rawData().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorLong(Tensor<TInt64> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new LongType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 8;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.rawData().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }
}
