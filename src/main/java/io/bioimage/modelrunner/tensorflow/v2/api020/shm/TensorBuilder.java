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

import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.ndarray.impl.buffer.raw.RawDataBufferFactory;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

/**
 * Utility class to build Tensorflow tensors from shm segments using {@link SharedMemoryArray}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder {

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	/**
	 * Creates {@link Tensor} instance from a {@link SharedMemoryArray}
	 * 
	 * @param array
	 * 	the {@link SharedMemoryArray} that is going to be converted into
	 *  a {@link Tensor} tensor
	 * @return the Tensorflow {@link Tensor} as the one stored in the shared memory segment
	 * @throws IllegalArgumentException if the type of the {@link SharedMemoryArray}
	 *  is not supported
	 */
	public static Tensor<? extends TType> build(SharedMemoryArray array) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (array.getOriginalDataType().equals("uint8")) {
			return buildUByte(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("int32")) {
			return buildInt(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("float32")) {
			return buildFloat(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("float64")) {
			return buildDouble(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("int64")) {
			return buildLong(Cast.unchecked(array));
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + array.getOriginalDataType());
		}
	}

	private static Tensor<TUint8> buildUByte(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		ByteDataBuffer dataBuffer = RawDataBufferFactory.create(buff.array(), false);
		Tensor<TUint8> ndarray = Tensor.of(TUint8.DTYPE, Shape.of(ogShape), dataBuffer);
		return ndarray;
	}

	private static Tensor<TInt32> buildInt(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		IntBuffer intBuff = buff.asIntBuffer();
		int[] intArray = new int[intBuff.capacity()];
		intBuff.get(intArray);
		IntDataBuffer dataBuffer = RawDataBufferFactory.create(intArray, false);
		Tensor<TInt32> ndarray = TInt32.tensorOf(Shape.of(ogShape), dataBuffer);
		return ndarray;
	}

	private static Tensor<TInt64> buildLong(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		LongBuffer longBuff = buff.asLongBuffer();
		long[] longArray = new long[longBuff.capacity()];
		longBuff.get(longArray);
		LongDataBuffer dataBuffer = RawDataBufferFactory.create(longArray, false);
		Tensor<TInt64> ndarray = TInt64.tensorOf(Shape.of(ogShape), dataBuffer);
		return ndarray;
	}

	private static Tensor<TFloat32> buildFloat(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		FloatBuffer floatBuff = buff.asFloatBuffer();
		float[] floatArray = new float[floatBuff.capacity()];
		floatBuff.get(floatArray);
		FloatDataBuffer dataBuffer = RawDataBufferFactory.create(floatArray, false);
		Tensor<TFloat32> ndarray = TFloat32.tensorOf(Shape.of(ogShape), dataBuffer);
		return ndarray;
	}

	private static Tensor<TFloat64> buildDouble(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		DoubleBuffer doubleBuff = buff.asDoubleBuffer();
		double[] doubleArray = new double[doubleBuff.capacity()];
		doubleBuff.get(doubleArray);
		DoubleDataBuffer dataBuffer = RawDataBufferFactory.create(doubleArray, false);
		Tensor<TFloat64> ndarray = TFloat64.tensorOf(Shape.of(ogShape), dataBuffer);
		return ndarray;
	}
}