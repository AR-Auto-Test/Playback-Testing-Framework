package c.e.b;

import android.hardware.SensorManager;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.Material;
import com.ibosoninnov.unitear.ARCoreSceneformActivity;
import java.util.function.Consumer;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class d implements Consumer {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ d f4619a = new d();

    @Override // java.util.function.Consumer
    public final void accept(Object obj) {
        SensorManager sensorManager = ARCoreSceneformActivity.r;
        ((Material) obj).setFloat3("color", new Color(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f));
    }
}