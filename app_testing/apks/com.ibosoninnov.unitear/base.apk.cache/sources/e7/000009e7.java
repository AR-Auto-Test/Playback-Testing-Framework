package c.e.b.ef;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.Filter;
import android.widget.Filterable;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import c.c.a.h;
import c.c.a.m.x.c.i;
import c.c.a.m.x.c.y;
import com.google.android.material.badge.BadgeDrawable;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* compiled from: ARGalleryAdapter.java */
/* loaded from: classes2.dex */
public class a extends RecyclerView.g<c> implements Filterable {

    /* renamed from: b  reason: collision with root package name */
    public List<c.e.b.hf.b> f4703b;

    /* renamed from: c  reason: collision with root package name */
    public List<c.e.b.hf.b> f4704c;

    /* renamed from: d  reason: collision with root package name */
    public List<c.e.b.hf.b> f4705d;

    /* renamed from: e  reason: collision with root package name */
    public Context f4706e;

    /* renamed from: f  reason: collision with root package name */
    public b f4707f;

    /* compiled from: ARGalleryAdapter.java */
    /* renamed from: c.e.b.ef.a$a  reason: collision with other inner class name */
    /* loaded from: classes2.dex */
    public class C0088a extends Filter {
        public C0088a() {
        }

        @Override // android.widget.Filter
        public Filter.FilterResults performFiltering(CharSequence charSequence) {
            String charSequence2 = charSequence.toString();
            if (charSequence2.isEmpty()) {
                a aVar = a.this;
                aVar.f4704c = aVar.f4703b;
                aVar.f4707f.a(false);
            } else {
                ArrayList arrayList = new ArrayList();
                ArrayList arrayList2 = new ArrayList();
                ArrayList arrayList3 = new ArrayList();
                a aVar2 = a.this;
                if (aVar2.f4705d != null) {
                    for (c.e.b.hf.b bVar : aVar2.f4703b) {
                        if (bVar.category.toLowerCase().contains(charSequence2.toLowerCase())) {
                            bVar.nameFilter = null;
                            arrayList.add(bVar);
                        }
                    }
                    for (c.e.b.hf.b bVar2 : a.this.f4705d) {
                        String[] split = bVar2.name.split(" ");
                        for (String str : split) {
                            String a2 = a.a(a.this, str);
                            if (str.toLowerCase().matches(charSequence2.toLowerCase()) || a2.toLowerCase().matches(charSequence2.toLowerCase())) {
                                arrayList2.add(a.b(a.this, bVar2));
                            }
                        }
                        if (arrayList2.size() == 0) {
                            for (String str2 : split) {
                                String a3 = a.a(a.this, str2);
                                if (str2.toLowerCase().startsWith(charSequence2.toLowerCase()) || a3.toLowerCase().startsWith(charSequence2.toLowerCase())) {
                                    arrayList3.add(a.b(a.this, bVar2));
                                }
                            }
                        }
                    }
                    arrayList.addAll(arrayList2);
                    if (arrayList2.size() == 0) {
                        arrayList.addAll(arrayList3);
                    }
                } else {
                    for (c.e.b.hf.b bVar3 : aVar2.f4703b) {
                        if (bVar3.name.toLowerCase().contains(charSequence2.toLowerCase())) {
                            arrayList.add(bVar3);
                        }
                    }
                }
                a.this.f4704c = arrayList;
                if (arrayList.size() == 0) {
                    a.this.f4707f.a(true);
                } else {
                    a.this.f4707f.a(false);
                }
            }
            Filter.FilterResults filterResults = new Filter.FilterResults();
            filterResults.values = a.this.f4704c;
            return filterResults;
        }

        @Override // android.widget.Filter
        public void publishResults(CharSequence charSequence, Filter.FilterResults filterResults) {
            a aVar = a.this;
            aVar.f4704c = (ArrayList) filterResults.values;
            aVar.notifyDataSetChanged();
        }
    }

    /* compiled from: ARGalleryAdapter.java */
    /* loaded from: classes2.dex */
    public interface b {
        void a(boolean z);
    }

    /* compiled from: ARGalleryAdapter.java */
    /* loaded from: classes2.dex */
    public static class c extends RecyclerView.d0 {

        /* renamed from: a  reason: collision with root package name */
        public final c.e.b.ff.c f4709a;

        public c(c.e.b.ff.c cVar) {
            super(cVar.B);
            this.f4709a = cVar;
        }
    }

    public a(List<c.e.b.hf.b> list, Context context, b bVar) {
        this.f4703b = list;
        this.f4704c = list;
        this.f4706e = context;
        this.f4707f = bVar;
        context.getResources().getDimension(R.dimen.argallery_thumbnail);
    }

    public static String a(a aVar, String str) {
        Objects.requireNonNull(aVar);
        str.hashCode();
        char c2 = 65535;
        switch (str.hashCode()) {
            case -1907941713:
                if (str.equals("People")) {
                    c2 = 0;
                    break;
                }
                break;
            case -959801728:
                if (str.equals("Analyses")) {
                    c2 = 1;
                    break;
                }
                break;
            case -686922873:
                if (str.equals("Indices")) {
                    c2 = 2;
                    break;
                }
                break;
            case 77238:
                if (str.equals("Men")) {
                    c2 = 3;
                    break;
                }
                break;
            case 2122698:
                if (str.equals("Data")) {
                    c2 = 4;
                    break;
                }
                break;
            case 2185678:
                if (str.equals("Feet")) {
                    c2 = 5;
                    break;
                }
                break;
            case 66226712:
                if (str.equals("Dozen")) {
                    c2 = 6;
                    break;
                }
                break;
            case 68241025:
                if (str.equals("Fungi")) {
                    c2 = 7;
                    break;
                }
                break;
            case 80685416:
                if (str.equals("Teeth")) {
                    c2 = '\b';
                    break;
                }
                break;
            case 81068824:
                if (str.equals("Trash")) {
                    c2 = '\t';
                    break;
                }
                break;
            case 83761118:
                if (str.equals("Women")) {
                    c2 = '\n';
                    break;
                }
                break;
            case 375442810:
                if (str.equals("Matrices")) {
                    c2 = 11;
                    break;
                }
                break;
            case 1499275331:
                if (str.equals("Settings")) {
                    c2 = '\f';
                    break;
                }
                break;
            case 1724170783:
                if (str.equals("Children")) {
                    c2 = '\r';
                    break;
                }
                break;
        }
        switch (c2) {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case '\b':
            case '\t':
            case '\n':
            case 11:
            case '\f':
            case '\r':
                str = "Settings";
                break;
        }
        if (str.endsWith("es")) {
            return str.substring(0, str.length() - 2) + CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
        } else if (str.endsWith("ies")) {
            return str.substring(0, str.length() - 2) + CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
        } else if (!str.endsWith("y")) {
            return str.endsWith("s") ? str.substring(0, str.length() - 1) : str;
        } else {
            return str.substring(0, str.length() - 3) + "ies";
        }
    }

    public static c.e.b.hf.b b(a aVar, c.e.b.hf.b bVar) {
        Objects.requireNonNull(aVar);
        c.e.b.hf.b bVar2 = new c.e.b.hf.b();
        bVar2.name = bVar.name;
        bVar2.imageUrl = bVar.imageUrl;
        bVar2.category = bVar.category;
        bVar2.id = bVar.id;
        bVar2.glbFile = bVar.glbFile;
        bVar2.nameFilter = CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
        return bVar2;
    }

    @Override // android.widget.Filterable
    public Filter getFilter() {
        return new C0088a();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.g
    public int getItemCount() {
        return this.f4704c.size();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [androidx.recyclerview.widget.RecyclerView$d0, int] */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public void onBindViewHolder(c cVar, int i) {
        c cVar2 = cVar;
        cVar2.f4709a.m(this.f4704c.get(i));
        if (this.f4705d != null) {
            if (this.f4704c.get(i).nameFilter == null && this.f4704c.get(i).imageUrl.size() > 1) {
                cVar2.f4709a.A.setText(this.f4704c.get(i).category);
                TextView textView = cVar2.f4709a.z;
                StringBuilder x = c.b.a.a.a.x("");
                x.append(this.f4704c.get(i).imageUrl.size());
                x.append(BadgeDrawable.DEFAULT_EXCEED_MAX_BADGE_NUMBER_SUFFIX);
                textView.setText(x.toString());
                c.c.a.b.e(this.f4706e).k(this.f4704c.get(i).imageUrl.get(0)).F((h) c.c.a.b.e(this.f4706e).j(Integer.valueOf((int) R.drawable.loading)).b()).t(new i(), new y(40)).B(cVar2.f4709a.s);
                if (this.f4704c.get(i).imageUrl.size() > 1) {
                    c.c.a.b.e(this.f4706e).k(this.f4704c.get(i).imageUrl.get(1)).F((h) c.c.a.b.e(this.f4706e).j(Integer.valueOf((int) R.drawable.loading)).b()).t(new i(), new y(40)).B(cVar2.f4709a.t);
                }
                if (this.f4704c.get(i).imageUrl.size() > 2) {
                    c.c.a.b.e(this.f4706e).k(this.f4704c.get(i).imageUrl.get(2)).F((h) c.c.a.b.e(this.f4706e).j(Integer.valueOf((int) R.drawable.loading)).b()).t(new i(), new y(40)).B(cVar2.f4709a.u);
                }
                if (this.f4704c.get(i).imageUrl.size() > 3) {
                    c.c.a.b.e(this.f4706e).k(this.f4704c.get(i).imageUrl.get(3)).F((h) c.c.a.b.e(this.f4706e).j(Integer.valueOf((int) R.drawable.loading)).b()).t(new i(), new y(40)).B(cVar2.f4709a.v);
                }
                cVar2.f4709a.r.setVisibility(8);
                cVar2.f4709a.w.setVisibility(0);
                return;
            }
            cVar2.f4709a.A.setText(this.f4704c.get(i).name);
            cVar2.f4709a.z.setText("");
            c.c.a.b.e(this.f4706e).k(this.f4704c.get(i).imageUrl.get(0)).F((h) c.c.a.b.e(this.f4706e).j(Integer.valueOf((int) R.drawable.loading)).b()).t(new i(), new y(40)).B(cVar2.f4709a.r);
            cVar2.f4709a.r.setVisibility(0);
            cVar2.f4709a.w.setVisibility(4);
            return;
        }
        cVar2.f4709a.A.setText(this.f4704c.get(i).name);
        cVar2.f4709a.z.setText("");
        c.c.a.b.e(this.f4706e).k(this.f4704c.get(i).imageUrl.get(0)).F((h) c.c.a.b.e(this.f4706e).j(Integer.valueOf((int) R.drawable.loading)).b()).t(new i(), new y(40)).B(cVar2.f4709a.r);
        cVar2.f4709a.r.setVisibility(0);
        cVar2.f4709a.w.setVisibility(4);
    }

    /* JADX DEBUG: Return type fixed from 'androidx.recyclerview.widget.RecyclerView$d0' to match base method */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public c onCreateViewHolder(ViewGroup viewGroup, int i) {
        c.e.b.ff.c cVar = (c.e.b.ff.c) b.m.f.b(LayoutInflater.from(viewGroup.getContext()), R.layout.item_ar_gallery, viewGroup, false);
        LinearLayout.LayoutParams layoutParams = (LinearLayout.LayoutParams) cVar.x.getLayoutParams();
        layoutParams.height = (int) (viewGroup.getWidth() / 4.5f);
        cVar.x.setLayoutParams(layoutParams);
        cVar.y.setLayoutParams(layoutParams);
        return new c(cVar);
    }

    public a(List<c.e.b.hf.b> list, List<c.e.b.hf.b> list2, Context context, b bVar) {
        this.f4703b = list;
        this.f4704c = list;
        this.f4705d = list2;
        this.f4706e = context;
        this.f4707f = bVar;
        context.getResources().getDimension(R.dimen.argallery_thumbnail);
    }
}