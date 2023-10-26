/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.2.0 API for Tensorflow 2.
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
package io.bioimage.modelrunner.tensorflow.v2.api020.tensor;


import io.bioimage.modelrunner.tensor.Utils;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

/**
 * A {@link RandomAccessibleInterval} builder for TensorFlow {@link Tensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Tensorflow 2 {@link Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class ImgLib2Builder
{
    /**
     * Utility class.
     */
    private ImgLib2Builder()
    {
    }

	/**    
	 * Creates a {@link RandomAccessibleInterval} from a given {@link Tensor} tensor
	 * @param <T>
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link Tensor} tensor data of datatype belonging to {@link TType} is read from.
	 * @return The {@link RandomAccessibleInterval} built from the {@link TType} tensor.
	 * @throws IllegalArgumentException If the {@link TType} tensor type is not supported.
	 */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> RandomAccessibleInterval<T> build(Tensor<? extends TType> tensor) throws IllegalArgumentException
    {
		switch (tensor.dataType().name())
        {
            case TUint8.NAME:
                return (RandomAccessibleInterval<T>) buildFromTensorUByte((Tensor<TUint8>) tensor);
            case TInt32.NAME:
                return (RandomAccessibleInterval<T>) buildFromTensorInt((Tensor<TInt32>) tensor);
            case TFloat32.NAME:
                return (RandomAccessibleInterval<T>) buildFromTensorFloat((Tensor<TFloat32>) tensor);
            case TFloat64.NAME:
                return (RandomAccessibleInterval<T>) buildFromTensorDouble((Tensor<TFloat64>) tensor);
            case TInt64.NAME:
                return (RandomAccessibleInterval<T>) buildFromTensorLong((Tensor<TInt64>) tensor);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned byte-typed {@link TUint8} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TUint8} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link UnsignedByteType}.
	 */
    private static RandomAccessibleInterval<UnsignedByteType> buildFromTensorUByte(Tensor<TUint8> tensor)
    {
    	long[] arrayShape = tensor.shape().asArray();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        byte[] flatArr = new byte[totalSize];
        tensor.rawData().read(flatArr);
		RandomAccessibleInterval<UnsignedByteType> rai = ArrayImgs.unsignedBytes(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned int32-typed {@link TInt32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt32} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link IntType}.
	 */
    private static RandomAccessibleInterval<IntType> buildFromTensorInt(Tensor<TInt32> tensor)
    {
    	long[] arrayShape = tensor.shape().asArray();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        int[] flatArr = new int[totalSize];
        tensor.rawData().asInts().read(flatArr);
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float32-typed {@link TFloat32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat32} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link FloatType}.
	 */
    private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(Tensor<TFloat32> tensor)
    {
    	long[] arrayShape = tensor.shape().asArray();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        float[] flatArr = new float[totalSize];
        tensor.rawData().asFloats().read(flatArr);
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float64-typed {@link TFloat64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat64} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link DoubleType}.
	 */
    private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(Tensor<TFloat64> tensor)
    {
    	long[] arrayShape = tensor.shape().asArray();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        double[] flatArr = new double[totalSize];
        tensor.rawData().asDoubles().read(flatArr);
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned int64-typed {@link TInt64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt64} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link LongType}.
	 */
    private static  RandomAccessibleInterval<LongType> buildFromTensorLong(Tensor<TInt64> tensor)
    {
    	long[] arrayShape = tensor.shape().asArray();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        long[] flatArr = new long[totalSize];
        tensor.rawData().asLongs().read(flatArr);
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return Utils.transpose(rai);
    }
}
