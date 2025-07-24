package c.e.b;

import android.content.DialogInterface;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class je implements DialogInterface.OnDismissListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f4954b;

    public je(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f4954b = nonARCoreActivitySceneform;
    }

    @Override // android.content.DialogInterface.OnDismissListener
    public void onDismiss(DialogInterface dialogInterface) {
        this.f4954b.U.setVisibility(0);
    }
}