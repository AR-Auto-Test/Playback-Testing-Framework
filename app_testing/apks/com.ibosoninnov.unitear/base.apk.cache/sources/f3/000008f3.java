package c.d.b.a;

import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.collision.Collider;
import com.google.ar.sceneform.utilities.EnvironmentalHdrParameters;
import java.util.function.BiConsumer;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class j implements BiConsumer {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ j f4306a = new j();

    @Override // java.util.function.BiConsumer
    public final void accept(Object obj, Object obj2) {
        EnvironmentalHdrParameters environmentalHdrParameters = Scene.DEFAULT_HDR_PARAMETERS;
        ((HitTestResult) obj).setNode((Node) ((Collider) obj2).getTransformProvider());
    }
}