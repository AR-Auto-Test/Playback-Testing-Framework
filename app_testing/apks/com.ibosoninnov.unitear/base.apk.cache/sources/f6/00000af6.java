package c.e.b;

import android.view.View;
import android.widget.Toast;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import com.ibosoninnov.unitear.R;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class re implements View.OnLongClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f5198b;

    public re(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f5198b = nonARCoreActivitySceneform;
    }

    @Override // android.view.View.OnLongClickListener
    public boolean onLongClick(View view) {
        NonARCoreActivitySceneform nonARCoreActivitySceneform = this.f5198b;
        if (nonARCoreActivitySceneform.X) {
            Toast.makeText(nonARCoreActivitySceneform.x, "Recording OFF", 1).show();
            NonARCoreActivitySceneform nonARCoreActivitySceneform2 = this.f5198b;
            nonARCoreActivitySceneform2.T.setText(nonARCoreActivitySceneform2.getResources().getString(R.string.photo_video));
        } else {
            Toast.makeText(nonARCoreActivitySceneform.x, "Recording ON", 1).show();
            NonARCoreActivitySceneform nonARCoreActivitySceneform3 = this.f5198b;
            nonARCoreActivitySceneform3.T.setText(nonARCoreActivitySceneform3.getResources().getString(R.string.stop_recording));
        }
        NonARCoreActivitySceneform.v(this.f5198b);
        return true;
    }
}