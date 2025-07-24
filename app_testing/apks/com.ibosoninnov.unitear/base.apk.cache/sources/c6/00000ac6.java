package c.e.b;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import java.util.Objects;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class oe implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f5108b;

    public oe(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f5108b = nonARCoreActivitySceneform;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        NonARCoreActivitySceneform nonARCoreActivitySceneform = this.f5108b;
        int i = NonARCoreActivitySceneform.r;
        Objects.requireNonNull(nonARCoreActivitySceneform);
        Log.d("NonARCoreActivity", "reload scene");
        new Intent(nonARCoreActivitySceneform, NonARCoreActivitySceneform.class).getComponent();
        Intent intent = nonARCoreActivitySceneform.getIntent();
        Bundle bundle = nonARCoreActivitySceneform.w0;
        if (bundle != null) {
            if (bundle.containsKey("alphaid")) {
                intent.putExtra("alphaid", nonARCoreActivitySceneform.w0.getString("alphaid"));
            }
            if (nonARCoreActivitySceneform.w0.containsKey("id")) {
                intent.putExtra("id", nonARCoreActivitySceneform.w0.getString("id"));
            }
            if (nonARCoreActivitySceneform.w0.containsKey("menuItemJson")) {
                intent.putExtra("menuItemJson", nonARCoreActivitySceneform.w0.getString("menuItemJson"));
            }
            if (nonARCoreActivitySceneform.w0.containsKey("groundContentId")) {
                intent.putExtra("groundContentId", nonARCoreActivitySceneform.w0.getString("groundContentId"));
            }
        }
        nonARCoreActivitySceneform.J.y();
        nonARCoreActivitySceneform.finish();
        nonARCoreActivitySceneform.startActivity(intent);
    }
}