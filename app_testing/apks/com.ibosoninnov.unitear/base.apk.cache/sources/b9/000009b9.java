package c.e.b;

import android.util.Log;
import c.e.b.jc;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import java.util.Objects;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class cb implements jc.c {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f4612a;

    public final void a(String str) {
        final NonARCoreActivitySceneform nonARCoreActivitySceneform = this.f4612a;
        Objects.requireNonNull(nonARCoreActivitySceneform);
        Log.d("NonARCoreActivity", "loaderARContentGroundPlane - " + str);
        nonARCoreActivitySceneform.q0.resetAnchor(0.5f, 0.5f);
        nonARCoreActivitySceneform.runOnUiThread(new Runnable() { // from class: c.e.b.eb
            @Override // java.lang.Runnable
            public final void run() {
                NonARCoreActivitySceneform.this.R.setVisibility(0);
            }
        });
    }
}