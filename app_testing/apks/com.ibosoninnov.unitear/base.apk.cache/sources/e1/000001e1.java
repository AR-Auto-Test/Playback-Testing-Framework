package b.b.h;

import android.annotation.SuppressLint;
import android.app.SearchableInfo;
import android.content.ComponentName;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.database.Cursor;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.TextUtils;
import android.text.style.TextAppearanceSpan;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.widget.SearchView;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.ibosoninnov.unitear.R;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.WeakHashMap;

/* compiled from: SuggestionsAdapter.java */
@SuppressLint({"RestrictedAPI"})
/* loaded from: classes.dex */
public class r0 extends b.k.a.c implements View.OnClickListener {
    public static final /* synthetic */ int m = 0;
    public final SearchView n;
    public final SearchableInfo o;
    public final Context p;
    public final WeakHashMap<String, Drawable.ConstantState> q;
    public final int r;
    public int s;
    public ColorStateList t;
    public int u;
    public int v;
    public int w;
    public int x;
    public int y;
    public int z;

    /* compiled from: SuggestionsAdapter.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final TextView f914a;

        /* renamed from: b  reason: collision with root package name */
        public final TextView f915b;

        /* renamed from: c  reason: collision with root package name */
        public final ImageView f916c;

        /* renamed from: d  reason: collision with root package name */
        public final ImageView f917d;

        /* renamed from: e  reason: collision with root package name */
        public final ImageView f918e;

        public a(View view) {
            this.f914a = (TextView) view.findViewById(16908308);
            this.f915b = (TextView) view.findViewById(16908309);
            this.f916c = (ImageView) view.findViewById(16908295);
            this.f917d = (ImageView) view.findViewById(16908296);
            this.f918e = (ImageView) view.findViewById(R.id.edit_query);
        }
    }

    public r0(Context context, SearchView searchView, SearchableInfo searchableInfo, WeakHashMap<String, Drawable.ConstantState> weakHashMap) {
        super(context, searchView.getSuggestionRowLayout(), null, true);
        this.s = 1;
        this.u = -1;
        this.v = -1;
        this.w = -1;
        this.x = -1;
        this.y = -1;
        this.z = -1;
        this.n = searchView;
        this.o = searchableInfo;
        this.r = searchView.getSuggestionCommitIconResId();
        this.p = context;
        this.q = weakHashMap;
    }

    public static String h(Cursor cursor, int i) {
        if (i == -1) {
            return null;
        }
        try {
            return cursor.getString(i);
        } catch (Exception e2) {
            Log.e("SuggestionsAdapter", "unexpected error retrieving valid column from cursor, did the remote process die?", e2);
            return null;
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r9v9, resolved type: android.text.SpannableString */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:59:0x0144  */
    /* JADX WARN: Removed duplicated region for block: B:60:0x0146  */
    @Override // b.k.a.a
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void a(View view, Context context, Cursor cursor) {
        Drawable f2;
        Drawable drawable;
        ActivityInfo activityInfo;
        int iconResource;
        String str;
        a aVar = (a) view.getTag();
        int i = this.z;
        int i2 = i != -1 ? cursor.getInt(i) : 0;
        if (aVar.f914a != null) {
            String h2 = h(cursor, this.u);
            TextView textView = aVar.f914a;
            textView.setText(h2);
            if (TextUtils.isEmpty(h2)) {
                textView.setVisibility(8);
            } else {
                textView.setVisibility(0);
            }
        }
        if (aVar.f915b != null) {
            String h3 = h(cursor, this.w);
            if (h3 != null) {
                if (this.t == null) {
                    TypedValue typedValue = new TypedValue();
                    this.f2307e.getTheme().resolveAttribute(R.attr.textColorSearchUrl, typedValue, true);
                    this.t = this.f2307e.getResources().getColorStateList(typedValue.resourceId);
                }
                SpannableString spannableString = new SpannableString(h3);
                spannableString.setSpan(new TextAppearanceSpan(null, 0, 0, this.t, null), 0, h3.length(), 33);
                str = spannableString;
            } else {
                str = h(cursor, this.v);
            }
            if (TextUtils.isEmpty(str)) {
                TextView textView2 = aVar.f914a;
                if (textView2 != null) {
                    textView2.setSingleLine(false);
                    aVar.f914a.setMaxLines(2);
                }
            } else {
                TextView textView3 = aVar.f914a;
                if (textView3 != null) {
                    textView3.setSingleLine(true);
                    aVar.f914a.setMaxLines(1);
                }
            }
            TextView textView4 = aVar.f915b;
            textView4.setText(str);
            if (TextUtils.isEmpty(str)) {
                textView4.setVisibility(8);
            } else {
                textView4.setVisibility(0);
            }
        }
        ImageView imageView = aVar.f916c;
        if (imageView != null) {
            int i3 = this.x;
            if (i3 == -1) {
                f2 = null;
            } else {
                f2 = f(cursor.getString(i3));
                if (f2 == null) {
                    ComponentName searchActivity = this.o.getSearchActivity();
                    String flattenToShortString = searchActivity.flattenToShortString();
                    if (this.q.containsKey(flattenToShortString)) {
                        Drawable.ConstantState constantState = this.q.get(flattenToShortString);
                        f2 = constantState == null ? null : constantState.newDrawable(this.p.getResources());
                    } else {
                        PackageManager packageManager = this.f2307e.getPackageManager();
                        try {
                            activityInfo = packageManager.getActivityInfo(searchActivity, 128);
                            iconResource = activityInfo.getIconResource();
                        } catch (PackageManager.NameNotFoundException e2) {
                            Log.w("SuggestionsAdapter", e2.toString());
                        }
                        if (iconResource != 0) {
                            drawable = packageManager.getDrawable(searchActivity.getPackageName(), iconResource, activityInfo.applicationInfo);
                            if (drawable == null) {
                                StringBuilder y = c.b.a.a.a.y("Invalid icon resource ", iconResource, " for ");
                                y.append(searchActivity.flattenToShortString());
                                Log.w("SuggestionsAdapter", y.toString());
                            }
                            this.q.put(flattenToShortString, drawable != null ? null : drawable.getConstantState());
                            f2 = drawable;
                        }
                        drawable = null;
                        this.q.put(flattenToShortString, drawable != null ? null : drawable.getConstantState());
                        f2 = drawable;
                    }
                    if (f2 == null) {
                        f2 = this.f2307e.getPackageManager().getDefaultActivityIcon();
                    }
                }
            }
            imageView.setImageDrawable(f2);
            if (f2 == null) {
                imageView.setVisibility(4);
            } else {
                imageView.setVisibility(0);
                f2.setVisible(false, false);
                f2.setVisible(true, false);
            }
        }
        ImageView imageView2 = aVar.f917d;
        if (imageView2 != null) {
            int i4 = this.y;
            Drawable f3 = i4 == -1 ? null : f(cursor.getString(i4));
            imageView2.setImageDrawable(f3);
            if (f3 == null) {
                imageView2.setVisibility(8);
            } else {
                imageView2.setVisibility(0);
                f3.setVisible(false, false);
                f3.setVisible(true, false);
            }
        }
        int i5 = this.s;
        if (i5 != 2 && (i5 != 1 || (i2 & 1) == 0)) {
            aVar.f918e.setVisibility(8);
            return;
        }
        aVar.f918e.setVisibility(0);
        aVar.f918e.setTag(aVar.f914a.getText());
        aVar.f918e.setOnClickListener(this);
    }

    @Override // b.k.a.a
    public void b(Cursor cursor) {
        try {
            super.b(cursor);
            if (cursor != null) {
                this.u = cursor.getColumnIndex("suggest_text_1");
                this.v = cursor.getColumnIndex("suggest_text_2");
                this.w = cursor.getColumnIndex("suggest_text_2_url");
                this.x = cursor.getColumnIndex("suggest_icon_1");
                this.y = cursor.getColumnIndex("suggest_icon_2");
                this.z = cursor.getColumnIndex("suggest_flags");
            }
        } catch (Exception e2) {
            Log.e("SuggestionsAdapter", "error changing cursor and caching columns", e2);
        }
    }

    @Override // b.k.a.a
    public CharSequence c(Cursor cursor) {
        String h2;
        String h3;
        if (cursor == null) {
            return null;
        }
        String h4 = h(cursor, cursor.getColumnIndex("suggest_intent_query"));
        if (h4 != null) {
            return h4;
        }
        if (!this.o.shouldRewriteQueryFromData() || (h3 = h(cursor, cursor.getColumnIndex("suggest_intent_data"))) == null) {
            if (!this.o.shouldRewriteQueryFromText() || (h2 = h(cursor, cursor.getColumnIndex("suggest_text_1"))) == null) {
                return null;
            }
            return h2;
        }
        return h3;
    }

    @Override // b.k.a.c, b.k.a.a
    public View d(Context context, Cursor cursor, ViewGroup viewGroup) {
        View inflate = this.l.inflate(this.j, viewGroup, false);
        inflate.setTag(new a(inflate));
        ((ImageView) inflate.findViewById(R.id.edit_query)).setImageResource(this.r);
        return inflate;
    }

    public Drawable e(Uri uri) {
        int parseInt;
        String authority = uri.getAuthority();
        if (!TextUtils.isEmpty(authority)) {
            try {
                Resources resourcesForApplication = this.f2307e.getPackageManager().getResourcesForApplication(authority);
                List<String> pathSegments = uri.getPathSegments();
                if (pathSegments != null) {
                    int size = pathSegments.size();
                    if (size == 1) {
                        try {
                            parseInt = Integer.parseInt(pathSegments.get(0));
                        } catch (NumberFormatException unused) {
                            throw new FileNotFoundException(c.b.a.a.a.n("Single path segment is not a resource ID: ", uri));
                        }
                    } else if (size == 2) {
                        parseInt = resourcesForApplication.getIdentifier(pathSegments.get(1), pathSegments.get(0), authority);
                    } else {
                        throw new FileNotFoundException(c.b.a.a.a.n("More than two path segments: ", uri));
                    }
                    if (parseInt != 0) {
                        return resourcesForApplication.getDrawable(parseInt);
                    }
                    throw new FileNotFoundException(c.b.a.a.a.n("No resource found for: ", uri));
                }
                throw new FileNotFoundException(c.b.a.a.a.n("No path: ", uri));
            } catch (PackageManager.NameNotFoundException unused2) {
                throw new FileNotFoundException(c.b.a.a.a.n("No package found for authority: ", uri));
            }
        }
        throw new FileNotFoundException(c.b.a.a.a.n("No authority: ", uri));
    }

    /* JADX WARN: Removed duplicated region for block: B:53:0x0133  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final Drawable f(String str) {
        Drawable drawable = null;
        if (str != null && !str.isEmpty() && !CrashlyticsReportDataCapture.SIGNAL_DEFAULT.equals(str)) {
            try {
                int parseInt = Integer.parseInt(str);
                String str2 = "android.resource://" + this.p.getPackageName() + "/" + parseInt;
                Drawable.ConstantState constantState = this.q.get(str2);
                Drawable newDrawable = constantState == null ? null : constantState.newDrawable();
                if (newDrawable != null) {
                    return newDrawable;
                }
                Context context = this.p;
                Object obj = b.j.c.a.f2074a;
                Drawable drawable2 = context.getDrawable(parseInt);
                if (drawable2 != null) {
                    this.q.put(str2, drawable2.getConstantState());
                }
                return drawable2;
            } catch (Resources.NotFoundException unused) {
                Log.w("SuggestionsAdapter", "Icon resource not found: " + str);
                return null;
            } catch (NumberFormatException unused2) {
                Drawable.ConstantState constantState2 = this.q.get(str);
                Drawable newDrawable2 = constantState2 == null ? null : constantState2.newDrawable();
                if (newDrawable2 != null) {
                    return newDrawable2;
                }
                Uri parse = Uri.parse(str);
                try {
                } catch (FileNotFoundException e2) {
                    Log.w("SuggestionsAdapter", "Icon not found: " + parse + ", " + e2.getMessage());
                }
                if ("android.resource".equals(parse.getScheme())) {
                    try {
                        drawable = e(parse);
                        if (drawable != null) {
                            this.q.put(str, drawable.getConstantState());
                        }
                    } catch (Resources.NotFoundException unused3) {
                        throw new FileNotFoundException("Resource does not exist: " + parse);
                    }
                } else {
                    InputStream openInputStream = this.p.getContentResolver().openInputStream(parse);
                    if (openInputStream != null) {
                        Drawable createFromStream = Drawable.createFromStream(openInputStream, null);
                        try {
                            openInputStream.close();
                        } catch (IOException e3) {
                            Log.e("SuggestionsAdapter", "Error closing icon stream for " + parse, e3);
                        }
                        drawable = createFromStream;
                        if (drawable != null) {
                        }
                    } else {
                        throw new FileNotFoundException("Failed to open " + parse);
                    }
                }
                Log.w("SuggestionsAdapter", "Icon not found: " + parse + ", " + e2.getMessage());
                if (drawable != null) {
                }
            }
        }
        return drawable;
    }

    public Cursor g(SearchableInfo searchableInfo, String str, int i) {
        String suggestAuthority;
        String[] strArr = null;
        if (searchableInfo == null || (suggestAuthority = searchableInfo.getSuggestAuthority()) == null) {
            return null;
        }
        Uri.Builder fragment = new Uri.Builder().scheme(FirebaseAnalytics.Param.CONTENT).authority(suggestAuthority).query("").fragment("");
        String suggestPath = searchableInfo.getSuggestPath();
        if (suggestPath != null) {
            fragment.appendEncodedPath(suggestPath);
        }
        fragment.appendPath("search_suggest_query");
        String suggestSelection = searchableInfo.getSuggestSelection();
        if (suggestSelection != null) {
            strArr = new String[]{str};
        } else {
            fragment.appendPath(str);
        }
        String[] strArr2 = strArr;
        if (i > 0) {
            fragment.appendQueryParameter("limit", String.valueOf(i));
        }
        return this.f2307e.getContentResolver().query(fragment.build(), null, suggestSelection, strArr2, null);
    }

    @Override // b.k.a.a, android.widget.BaseAdapter, android.widget.SpinnerAdapter
    public View getDropDownView(int i, View view, ViewGroup viewGroup) {
        try {
            return super.getDropDownView(i, view, viewGroup);
        } catch (RuntimeException e2) {
            Log.w("SuggestionsAdapter", "Search suggestions cursor threw exception.", e2);
            View inflate = this.l.inflate(this.k, viewGroup, false);
            if (inflate != null) {
                ((a) inflate.getTag()).f914a.setText(e2.toString());
            }
            return inflate;
        }
    }

    @Override // b.k.a.a, android.widget.Adapter
    public View getView(int i, View view, ViewGroup viewGroup) {
        try {
            return super.getView(i, view, viewGroup);
        } catch (RuntimeException e2) {
            Log.w("SuggestionsAdapter", "Search suggestions cursor threw exception.", e2);
            View d2 = d(this.f2307e, this.f2306d, viewGroup);
            ((a) d2.getTag()).f914a.setText(e2.toString());
            return d2;
        }
    }

    @Override // android.widget.BaseAdapter, android.widget.Adapter
    public boolean hasStableIds() {
        return false;
    }

    public final void i(Cursor cursor) {
        Bundle extras = cursor != null ? cursor.getExtras() : null;
        if (extras == null || extras.getBoolean("in_progress")) {
        }
    }

    @Override // android.widget.BaseAdapter
    public void notifyDataSetChanged() {
        super.notifyDataSetChanged();
        i(this.f2306d);
    }

    @Override // android.widget.BaseAdapter
    public void notifyDataSetInvalidated() {
        super.notifyDataSetInvalidated();
        i(this.f2306d);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        Object tag = view.getTag();
        if (tag instanceof CharSequence) {
            this.n.k((CharSequence) tag);
        }
    }
}