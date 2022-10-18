package plugins.danyfel80.tensorflow.v2.api020.test;

import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.types.TInt32;

import icy.main.Icy;
import icy.plugin.PluginLauncher;
import icy.plugin.PluginLoader;
import plugins.adufour.ezplug.EzPlug;

/**
 * @author Daniel Felipe Gonzalez Obando
 */
public class TensorFlowTestPlugin extends EzPlug
{
    private static Signature dbl(Ops tf)
    {
        Placeholder<TInt32> x = tf.placeholder(TInt32.DTYPE);
        Add<TInt32> dblX = tf.math.add(x, x);
        return Signature.builder().input("x", x).output("dbl", dblX).build();
    }

    @Override
    public void clean()
    {
    }

    @Override
    protected void execute()
    {

        System.out.println("Hello TensorFlow " + TensorFlow.version());

        try (ConcreteFunction dbl = ConcreteFunction.create(TensorFlowTestPlugin::dbl);
                Tensor<TInt32> x = TInt32.scalarOf(10);
                Tensor<TInt32> dblX = dbl.call(x).expect(TInt32.DTYPE))
        {
            System.out.println(x.data().getInt() + " doubled is " + dblX.data().getInt());
        }

    }

    @Override
    protected void initialize()
    {
    }

    public static void main(String[] args)
    {
        Icy.main(args);
        PluginLauncher.start(PluginLoader.getPlugin(TensorFlowTestPlugin.class.getName()));

    }

}
