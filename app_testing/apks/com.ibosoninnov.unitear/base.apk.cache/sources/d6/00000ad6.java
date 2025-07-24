package c.e.b;

import android.view.View;
import android.widget.Toast;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class pe implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f5141b;

    public pe(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f5141b = nonARCoreActivitySceneform;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        NonARCoreActivitySceneform nonARCoreActivitySceneform = this.f5141b;
        if (nonARCoreActivitySceneform.X) {
            Toast.makeText(nonARCoreActivitySceneform.x, "Recording OFF", 1).show();
            NonARCoreActivitySceneform.v(this.f5141b);
            return;
        }
        nonARCoreActivitySceneform.D.setParent(null);
        this.f5141b.finish();
    }
}