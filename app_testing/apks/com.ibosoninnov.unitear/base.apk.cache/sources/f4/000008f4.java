package c.d.b.a;

import android.util.Log;
import com.google.ar.sceneform.Scene;
import java.util.function.Function;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class k implements Function {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ k f4307a = new k();

    @Override // java.util.function.Function
    public final Object apply(Object obj) {
        Log.e(Scene.TAG, "Failed to create the default Light Probe: ", (Throwable) obj);
        return null;
    }
}