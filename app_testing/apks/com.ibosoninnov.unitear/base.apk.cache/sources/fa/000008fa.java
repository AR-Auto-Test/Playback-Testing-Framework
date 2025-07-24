package c.d.b.a.q;

import android.util.Log;
import com.google.ar.sceneform.rendering.CameraStream;
import java.util.function.Function;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class a implements Function {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4313a = new a();

    @Override // java.util.function.Function
    public final Object apply(Object obj) {
        Log.e(CameraStream.TAG, "Unable to load camera stream materials.", (Throwable) obj);
        return null;
    }
}