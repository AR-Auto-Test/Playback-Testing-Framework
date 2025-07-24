package c.d.b.a.q;

import android.util.Log;
import com.google.ar.sceneform.rendering.PlaneRenderer;
import java.util.function.Function;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class a0 implements Function {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a0 f4314a = new a0();

    @Override // java.util.function.Function
    public final Object apply(Object obj) {
        Log.e(PlaneRenderer.TAG, "Unable to load plane shadow material.", (Throwable) obj);
        return null;
    }
}