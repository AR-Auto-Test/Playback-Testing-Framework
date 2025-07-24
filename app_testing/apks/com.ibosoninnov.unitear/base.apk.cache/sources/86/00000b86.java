package c.e.b;

import android.util.Log;
import c.e.b.ec;
import c.e.b.vc;
import com.google.ar.core.InstallActivity;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import java.util.Objects;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class zc implements ec.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ vc.c f5513a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ vc f5514b;

    public zc(vc vcVar, vc.c cVar) {
        this.f5514b = vcVar;
        this.f5513a = cVar;
    }

    @Override // c.e.b.ec.a
    public void a(String str) {
    }

    @Override // c.e.b.ec.a
    public void b(String str) {
        Objects.requireNonNull(this.f5514b);
        Log.d("LoaderARContentGroundPlaneSceneformARCore", str);
        try {
            JSONObject jSONObject = new JSONObject(str);
            if (jSONObject.getBoolean(SettingsJsonConstants.APP_STATUS_KEY)) {
                JSONArray jSONArray = jSONObject.getJSONObject("data").getJSONObject("arBundle").getJSONArray("arContent");
                for (int i = 0; i < jSONArray.length(); i++) {
                    vc.b(this.f5514b, jSONArray.getJSONObject(i), i);
                }
                Objects.requireNonNull(((g) this.f5513a).f4760a);
                Log.d("ARCoreSceneformActivity", "");
                return;
            }
            ((g) this.f5513a).a(jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY));
        } catch (JSONException e2) {
            e2.printStackTrace();
            ((g) this.f5513a).a(e2.getMessage());
        }
    }
}