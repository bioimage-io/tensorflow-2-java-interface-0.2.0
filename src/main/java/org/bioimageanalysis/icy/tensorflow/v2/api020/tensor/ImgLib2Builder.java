package org.bioimageanalysis.icy.tensorflow.v2.api020.tensor;

import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;


/**
 * @author Carlos GArcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class ImgLib2Builder
{
    /**
     * Utility class.
     */
    private ImgLib2Builder()
    {
    }

    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(Tensor<? extends TType> tensor) throws IllegalArgumentException
    {
		switch (tensor.dataType().name())
        {
            case TUint8.NAME:
                return (Img<T>) buildFromTensorByte((Tensor<TUint8>) tensor);
            case TInt32.NAME:
                return (Img<T>) buildFromTensorInt((Tensor<TInt32>) tensor);
            case TFloat32.NAME:
                return (Img<T>) buildFromTensorFloat((Tensor<TFloat32>) tensor);
            case TFloat64.NAME:
                return (Img<T>) buildFromTensorDouble((Tensor<TFloat64>) tensor);
            case TInt64.NAME:
                return (Img<T>) buildFromTensorLong((Tensor<TInt64>) tensor);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

    private static ArrayImg<ByteType, ByteArray> buildFromTensorByte(Tensor<TUint8> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		byte[] flatImageArray = new byte[(int) size];
		// Copy data from tensor to array
        tensor.rawData().read(flatImageArray);
		return ArrayImgs.bytes(flatImageArray, tensorShape);
    }

    private static ArrayImg<IntType, IntArray> buildFromTensorInt(Tensor<TInt32> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		int[] flatImageArray = new int[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asInts().read(flatImageArray);
		return ArrayImgs.ints(flatImageArray, tensorShape);
    }

    private static <T extends Type<T>> ArrayImg<FloatType, FloatArray> buildFromTensorFloat(Tensor<TFloat32> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		float[] flatImageArray = new float[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asFloats().read(flatImageArray);
		return ArrayImgs.floats(flatImageArray, tensorShape);
    }

    private static <T extends Type<T>> ArrayImg<DoubleType, DoubleArray> buildFromTensorDouble(Tensor<TFloat64> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		double[] flatImageArray = new double[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asDoubles().read(flatImageArray);
		return ArrayImgs.doubles(flatImageArray, tensorShape);
    }

    private static  ArrayImg<LongType, LongArray> buildFromTensorLong(Tensor<TInt64> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		long[] flatImageArray = new long[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asLongs().read(flatImageArray);
		return ArrayImgs.longs(flatImageArray, tensorShape);
    }
}
