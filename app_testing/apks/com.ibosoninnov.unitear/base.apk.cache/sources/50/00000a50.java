package c.e.b;

import android.view.View;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class ie implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f4861b;

    public ie(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f4861b = nonARCoreActivitySceneform;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        this.f4861b.d0.dismiss();
    }
}