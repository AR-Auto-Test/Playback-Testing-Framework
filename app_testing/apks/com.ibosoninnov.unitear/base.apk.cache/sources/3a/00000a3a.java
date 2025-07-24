package c.e.b.hf;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.util.ArrayList;
import java.util.List;

/* compiled from: ARGalleryItem.java */
/* loaded from: classes2.dex */
public class a {
    public String category;
    public String file_loc;
    public String id;
    public String prefab_name;
    public String thumbnail_url;
    public boolean isLoaded = false;
    public int downloadStatus = -1;

    /* compiled from: ARGalleryItem.java */
    /* renamed from: c.e.b.hf.a$a  reason: collision with other inner class name */
    /* loaded from: classes2.dex */
    public class C0089a extends TypeToken<List<a>> {
    }

    public static ArrayList<a> a(String str) {
        return (ArrayList) new Gson().fromJson(str, new C0089a().getType());
    }
}