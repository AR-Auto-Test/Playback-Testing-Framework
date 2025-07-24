package c.e.b;

import android.content.DialogInterface;
import com.ibosoninnov.unitear.ARCoreSceneformActivity;

/* compiled from: ARCoreSceneformActivity.java */
/* loaded from: classes2.dex */
public class sb implements DialogInterface.OnDismissListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ARCoreSceneformActivity f5222b;

    public sb(ARCoreSceneformActivity aRCoreSceneformActivity) {
        this.f5222b = aRCoreSceneformActivity;
    }

    @Override // android.content.DialogInterface.OnDismissListener
    public void onDismiss(DialogInterface dialogInterface) {
        this.f5222b.R.setVisibility(0);
        ARCoreSceneformActivity aRCoreSceneformActivity = this.f5222b;
        if (!aRCoreSceneformActivity.E) {
            aRCoreSceneformActivity.A(true, false);
            this.f5222b.d0.setVisibility(0);
            return;
        }
        aRCoreSceneformActivity.c0.setVisibility(0);
        this.f5222b.Z.setVisibility(0);
    }
}