package b.j.g;

import android.content.ContentUris;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.pm.ProviderInfo;
import android.content.pm.Signature;
import android.content.res.Resources;
import android.database.Cursor;
import android.net.Uri;
import android.os.CancellationSignal;
import com.google.firebase.analytics.FirebaseAnalytics;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/* compiled from: FontProvider.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public static final Comparator<byte[]> f2128a = new a();

    /* compiled from: FontProvider.java */
    /* loaded from: classes.dex */
    public class a implements Comparator<byte[]> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(byte[] bArr, byte[] bArr2) {
            int i;
            int i2;
            byte[] bArr3 = bArr;
            byte[] bArr4 = bArr2;
            if (bArr3.length != bArr4.length) {
                i = bArr3.length;
                i2 = bArr4.length;
            } else {
                for (int i3 = 0; i3 < bArr3.length; i3++) {
                    if (bArr3[i3] != bArr4[i3]) {
                        i = bArr3[i3];
                        i2 = bArr4[i3];
                    }
                }
                return 0;
            }
            return i - i2;
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:30:0x0090 A[LOOP:1: B:15:0x004b->B:30:0x0090, LOOP_END] */
    /* JADX WARN: Removed duplicated region for block: B:80:0x0094 A[EDGE_INSN: B:80:0x0094->B:32:0x0094 ?: BREAK  , SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static k a(Context context, e eVar, CancellationSignal cancellationSignal) {
        Cursor cursor;
        Uri withAppendedId;
        boolean z;
        PackageManager packageManager = context.getPackageManager();
        Resources resources = context.getResources();
        String str = eVar.f2129a;
        ProviderInfo resolveContentProvider = packageManager.resolveContentProvider(str, 0);
        if (resolveContentProvider != null) {
            if (resolveContentProvider.packageName.equals(eVar.f2130b)) {
                Signature[] signatureArr = packageManager.getPackageInfo(resolveContentProvider.packageName, 64).signatures;
                ArrayList arrayList = new ArrayList();
                for (Signature signature : signatureArr) {
                    arrayList.add(signature.toByteArray());
                }
                Collections.sort(arrayList, f2128a);
                List<List<byte[]>> list = eVar.f2132d;
                if (list == null) {
                    list = b.j.b.d.L(resources, 0);
                }
                int i = 0;
                while (true) {
                    cursor = null;
                    if (i >= list.size()) {
                        resolveContentProvider = null;
                        break;
                    }
                    ArrayList arrayList2 = new ArrayList(list.get(i));
                    Collections.sort(arrayList2, f2128a);
                    if (arrayList.size() == arrayList2.size()) {
                        for (int i2 = 0; i2 < arrayList.size(); i2++) {
                            if (Arrays.equals((byte[]) arrayList.get(i2), (byte[]) arrayList2.get(i2))) {
                            }
                        }
                        z = true;
                        if (!z) {
                            break;
                        }
                        i++;
                    }
                    z = false;
                    if (!z) {
                    }
                }
                if (resolveContentProvider == null) {
                    return new k(1, null);
                }
                String str2 = resolveContentProvider.authority;
                ArrayList arrayList3 = new ArrayList();
                Uri build = new Uri.Builder().scheme(FirebaseAnalytics.Param.CONTENT).authority(str2).build();
                Uri build2 = new Uri.Builder().scheme(FirebaseAnalytics.Param.CONTENT).authority(str2).appendPath("file").build();
                try {
                    cursor = context.getContentResolver().query(build, new String[]{"_id", "file_id", "font_ttc_index", "font_variation_settings", "font_weight", "font_italic", "result_code"}, "query = ?", new String[]{eVar.f2131c}, null, null);
                    if (cursor != null && cursor.getCount() > 0) {
                        int columnIndex = cursor.getColumnIndex("result_code");
                        arrayList3 = new ArrayList();
                        int columnIndex2 = cursor.getColumnIndex("_id");
                        int columnIndex3 = cursor.getColumnIndex("file_id");
                        int columnIndex4 = cursor.getColumnIndex("font_ttc_index");
                        int columnIndex5 = cursor.getColumnIndex("font_weight");
                        int columnIndex6 = cursor.getColumnIndex("font_italic");
                        while (cursor.moveToNext()) {
                            int i3 = columnIndex != -1 ? cursor.getInt(columnIndex) : 0;
                            int i4 = columnIndex4 != -1 ? cursor.getInt(columnIndex4) : 0;
                            if (columnIndex3 == -1) {
                                withAppendedId = ContentUris.withAppendedId(build, cursor.getLong(columnIndex2));
                            } else {
                                withAppendedId = ContentUris.withAppendedId(build2, cursor.getLong(columnIndex3));
                            }
                            arrayList3.add(new l(withAppendedId, i4, columnIndex5 != -1 ? cursor.getInt(columnIndex5) : 400, columnIndex6 != -1 && cursor.getInt(columnIndex6) == 1, i3));
                        }
                    }
                    return new k(0, (l[]) arrayList3.toArray(new l[0]));
                } finally {
                    if (cursor != null) {
                        cursor.close();
                    }
                }
            }
            StringBuilder B = c.b.a.a.a.B("Found content provider ", str, ", but package was not ");
            B.append(eVar.f2130b);
            throw new PackageManager.NameNotFoundException(B.toString());
        }
        throw new PackageManager.NameNotFoundException(c.b.a.a.a.q("No package found for authority: ", str));
    }
}