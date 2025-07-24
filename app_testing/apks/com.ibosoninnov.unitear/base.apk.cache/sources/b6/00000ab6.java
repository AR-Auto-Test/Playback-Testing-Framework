package c.e.b;

import android.content.Intent;
import android.view.View;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import com.ibosoninnov.unitear.activities.Help2Activity;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class ne implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f5074b;

    public ne(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f5074b = nonARCoreActivitySceneform;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        this.f5074b.startActivity(new Intent(this.f5074b, Help2Activity.class));
    }
}