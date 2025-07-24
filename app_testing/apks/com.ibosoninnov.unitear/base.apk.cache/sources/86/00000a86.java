package c.e.b;

import android.content.DialogInterface;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class ke implements DialogInterface.OnDismissListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f4981b;

    public ke(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f4981b = nonARCoreActivitySceneform;
    }

    @Override // android.content.DialogInterface.OnDismissListener
    public void onDismiss(DialogInterface dialogInterface) {
        this.f4981b.N.setVisibility(0);
        this.f4981b.M.setVisibility(0);
        this.f4981b.U.setVisibility(0);
        this.f4981b.R.setVisibility(0);
    }
}