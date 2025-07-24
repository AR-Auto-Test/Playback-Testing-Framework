package c.d.b.a;

import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.utilities.EnvironmentalHdrParameters;
import java.util.function.Supplier;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class m implements Supplier {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ m f4309a = new m();

    @Override // java.util.function.Supplier
    public final Object get() {
        EnvironmentalHdrParameters environmentalHdrParameters = Scene.DEFAULT_HDR_PARAMETERS;
        return new HitTestResult();
    }
}