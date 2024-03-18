MITOTIC_STATUS_CODE = 7
MITOTIC_OR_APOPTOTIC_CODE = 5
NORMAL_CODE = 4


# Extracting lineage-centric information
def extract_lineage_centric_info(fs_nodes):
    lineage_data = []
    mitosis_count = 0
    for fs in fs_nodes:
        for f in fs.findall("f"):
            for _as in f.findall("as"):
                for a_node in _as.findall(".//a"):
                    cell_id = a_node.get("id")
                    parent_cell_id = f.get("id")  # Parent cell ID is now the ID of 'f' node
                    sub_as_elements = a_node.findall("as")
                    if len(sub_as_elements) == 0:  # Cell without daughters
                        continue
                    sub_as_node = sub_as_elements[0]
                    sub_as_a_elements = sub_as_node.findall("a")
                    daughtercell_ids = []
                    if len(sub_as_a_elements) == 2:  # Cell with daughters
                        for as_element in sub_as_elements:
                            for daughter_cell in as_element.findall("a"):
                                daughtercell_ids.append(daughter_cell.get("id"))
                        mitosis_count += 1
                    elif len(sub_as_a_elements) > 2:
                        print(f"Error: More than 2 'as' elements for cell ID {cell_id}")
                    lineage_data.append(
                        {"cellID": cell_id, "parentcellID": parent_cell_id, "daughtercellIDs": daughtercell_ids}
                    )
    print("Number of mitosis events:", mitosis_count)
    return lineage_data


# Extracting frame-centric information
def extract_frame_centric_info(fs_nodes):
    frame_data = []
    for fs in fs_nodes:
        for f in fs.findall("f"):
            for _as in f.findall("as"):
                for a in _as.findall(".//a"):
                    for ss in a.findall("ss"):
                        cell_id = a.get("id")  # Cell ID is now the ID of 'a' node
                        cell_color = a.get("brush")
                        cell_type = a.get("type")
                        xs, ys = [], []
                        cell_status = []
                        timepoints = []
                        for s in ss.findall("s"):
                            # print("--flag1")
                            xcoord = float(s.get("x"))
                            ycoord = float(s.get("y"))
                            _cs = int(s.get("s"))
                            xs.append(xcoord)
                            ys.append(ycoord)
                            cell_status.append(_cs)
                            timepoints.append(s.get("i"))

                        frame_data.append(
                            {
                                "cellID": cell_id,
                                "cellColour": cell_color,
                                "cellType": cell_type,
                                "xcoords": xs,
                                "ycoords": ys,
                                "cellStatus": cell_status,
                                "timepoints": timepoints,
                            }
                        )
    return frame_data
