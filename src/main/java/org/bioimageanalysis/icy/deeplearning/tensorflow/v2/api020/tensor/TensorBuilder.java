package org.bioimageanalysis.icy.deeplearning.tensorflow.v2.api020.tensor;

import org.bioimageanalysis.icy.deeplearning.utils.IndexingUtils;
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

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;


/**
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class TensorBuilder
{

    /**
     * Utility class.
     */
    private TensorBuilder()
    {
    }

    /**
     * Creates {@link Tensor} instance with the same size and information as the given {@link RandomAccessibleInerval}.
     * 
     * @param sequence
     *        The sequence which the created tensor is filled with.
     * @return The created tensor.
     * @throws IllegalArgumentException
     *         If the type of the sequence is not supported.
     */
    public static Tensor<? extends TType> build(org.bioimageanalysis.icy.deeplearning.tensor.Tensor tensor) throws IllegalArgumentException
    {
    	return build(tensor.getData());
    }

    /**
     * Creates {@link Tensor} instance with the same size and information as the given {@link RandomAccessibleInerval}.
     * 
     * @param sequence
     *        The sequence which the created tensor is filled with.
     * @return The created tensor.
     * @throws IllegalArgumentException
     *         If the type of the sequence is not supported.
     */
    public static <T extends Type<T>> Tensor<? extends TType> build(RandomAccessibleInterval<T> array) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
    	if (Util.getTypeFromInterval(array) instanceof ByteType) {
            return buildByte((RandomAccessibleInterval<ByteType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof IntType) {
            return buildInt((RandomAccessibleInterval<IntType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof FloatType) {
            return buildFloat((RandomAccessibleInterval<FloatType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof DoubleType) {
            return buildDouble((RandomAccessibleInterval<DoubleType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof LongType) {
            return buildLong((RandomAccessibleInterval<LongType>) array);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + Util.getTypeFromInterval(array).getClass().toString());
    	}
    }

    /**
     * Creates a {@link Tensor} of type {@link TUint8} from an {@link RandomAccessibleInterval} of type {@link DataType#BYTE} 
     * or {@link DataType#UBYTE}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static Tensor<TUint8> buildByte(RandomAccessibleInterval<ByteType> imgTensor) throws IllegalArgumentException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<ByteType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<ByteType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		byte[] flatArr = new byte[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = tensorCursor.get().getByte();
        	flatArr[flatPos] = val;
		}
        ByteDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		Tensor<TUint8> tensor = Tensor.of(TUint8.DTYPE, Shape.of(imgTensor.dimensionsAsLongArray()), dataBuffer);
	 	return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TInt32} from an {@link RandomAccessibleInterval} of type {@link DataType#INT} or
     * {@link DataType#UINT}.
     * 
     * @param sequence
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static Tensor<TInt32> buildInt(RandomAccessibleInterval<IntType> imgTensor) throws IllegalArgumentException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<IntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<IntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		int[] flatArr = new int[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = tensorCursor.get().getInt();
        	flatArr[flatPos] = val;
		}
		IntDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		Tensor<TInt32> tensor = TInt32.tensorOf(Shape.of(imgTensor.dimensionsAsLongArray()), dataBuffer);
	 	return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TInt64} from an {@link RandomAccessibleInterval} of type {@link DataType#LONG}
     * 
     * @param sequence
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static Tensor<TInt64> buildLong(RandomAccessibleInterval<LongType> imgTensor) throws IllegalArgumentException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<LongType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<LongType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<LongType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		long[] flatArr = new long[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = tensorCursor.get().getLong();
        	flatArr[flatPos] = val;
		}
		LongDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		Tensor<TInt64> tensor = TInt64.tensorOf(Shape.of(imgTensor.dimensionsAsLongArray()), dataBuffer);
	 	return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TFloat32} from an {@link RandomAccessibleInterval} of type {@link DataType#FLOAT}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static Tensor<TFloat32> buildFloat(RandomAccessibleInterval<FloatType> imgTensor) throws IllegalArgumentException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<FloatType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<FloatType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		float[] flatArr = new float[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = tensorCursor.get().getRealFloat();
        	flatArr[flatPos] = val;
		}
		FloatDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		Tensor<TFloat32> tensor = TFloat32.tensorOf(Shape.of(imgTensor.dimensionsAsLongArray()), dataBuffer);
	 	return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TFloat64} from an {@link RandomAccessibleInterval} of type {@link DataType#DOUBLE}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static Tensor<TFloat64> buildDouble(RandomAccessibleInterval<DoubleType> imgTensor) throws IllegalArgumentException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<DoubleType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		double[] flatArr = new double[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = tensorCursor.get().getRealDouble();
        	flatArr[flatPos] = val;
		}
		DoubleDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		Tensor<TFloat64> tensor = TFloat64.tensorOf(Shape.of(imgTensor.dimensionsAsLongArray()), dataBuffer);
	 	return tensor;
    }
}