// Data objects

class QCDash_Subject {

	constructor(p_numericID, p_textID, 
				p_t1CorrTarget, p_t1OverlapBrain, 
				p_boldCorrTarget, p_boldOverlapBrain,
				p_listIndices) {

		this.numericID = p_numericID;
		this.textID = p_textID;
		this.t1 = { corrTarget: p_t1CorrTarget, overlapBrain: p_t1OverlapBrain };
		this.bold = { corrTarget: p_boldCorrTarget, overlapBrain: p_boldOverlapBrain };
		this.listIndices = p_listIndices;
	}

	indexInChart(p_chartID) {

		return this.listIndices[p_chartID];
	}
}

class QCDash_Run {

	constructor(p_numericID, p_textID, 
				p_intraCorrTarget, 
				p_fdBefore, p_fdAfter, 
				p_volScrubbed, p_volOk,
				p_listIndices) {

		this.numericID = p_numericID;
		this.textID = p_textID;
		this.intra = { corrTarget: p_intraCorrTarget };
		this.fd = { before: p_fdBefore, after: p_fdAfter };
		this.nbVol = { scrubbed: p_volScrubbed, ok: p_volOk };
		this.listIndices = p_listIndices;
	}

	indexInChart(p_chartID) {

		return this.listIndices[p_chartID];
	}

}


// Keeps track and messages charts currently visible on the QC dashboard
class QCDash_Charts {

	constructor(p_dataDictionary, p_chartTitleDivID, p_chartNotesDivID) {

		this.m_chartDictionary = {};
		this.m_dataDictionary = p_dataDictionary;
        this.m_currentChart = null;

        this.m_chartTitleDivID = p_chartTitleDivID;
        this.m_chartNotesDivID = p_chartNotesDivID;
    }

	addChart(p_chartID, p_data, p_chartObject){

		// Add this chart and its data (indices 1 and 0 in the chart dictionary, respectively)
		this.m_chartDictionary[p_chartID] = [];
		this.m_chartDictionary[p_chartID].push(p_data);
		this.m_chartDictionary[p_chartID].push(p_chartObject);
	}

    generateChart(p_chartID) {

        if ( p_chartID != this.m_currentChart ) {

            let myChart = this.m_chartDictionary[p_chartID][1];
            
            // Change chart div title and notes
            $("#" + this.m_chartTitleDivID).html(myChart.m_chartTitle);
            $("#" + this.m_chartNotesDivID).html(myChart.m_chartNotes);

            // Generate the chart
            this.m_chartDictionary[p_chartID][1].generate();

            // Save this chart ID as the current one generated
            this.m_currentChart = p_chartID;
        }
    }

	generateCharts() {

        for ( var chartID in this.m_chartDictionary ) {
            
            this.m_chartDictionary[chartID][1].generate();
        }		
	}

    selectChartData(p_dataPointName){

        // Show tooltip for data point
        if ( null !== this.m_currentChart ) {

            let dataIndex = this.m_dataDictionary[p_dataPointName].indexInChart(this.m_currentChart);
            this.m_chartDictionary[this.m_currentChart][1].m_chartRef.tooltip.show({index: dataIndex - 1});
            this.m_chartDictionary[this.m_currentChart][1].m_chartRef.select([this.m_chartDictionary[this.m_currentChart][0][0][0]], [dataIndex - 1]);
        }
    }    

	// signalChartsFocus(p_dataPointName, p_chartIDs){

	// 	// Lookup datapoint to focus on
	// 	for ( let index = 0; index < p_chartIDs.length; index++ ) {

	// 		// Show tooltip for data point
	// 		let dataIndex = this.m_dataDictionary[p_dataPointName].indexInChart(p_chartIDs[index]);
	// 		this.m_chartDictionary[p_chartIDs[index]][1].m_chartRef.tooltip.show({index: dataIndex - 1});
	// 	}
	// }

}

// Data functions

function getIndexOfKeyInList(p_key, p_list) {

	let foundIndex = -1;

	for ( let index = 0; index < p_list.length; index++ ){

		if ( p_key == p_list[index] ) {
			foundIndex = index;
			break;
		}
	}

	return foundIndex;
}

function createSubjectDictionary(dataT1, dataOverlapT1, dataBold, dataBrain) {

	var subjectTable = {};

    // Create subject list based on dataTI[0][1...n]
    // NOTE: This works for now, but maybe a cleaner source for subject IDs can be created
    // J.Armoza - 6/27/18
    let listSubjects = [];
    for ( let index = 1; index < dataT1[0].length; index++ ){
        listSubjects.push({ id: index, text: dataT1[0][index] });
    }

	// Create objects for each subject
	for ( let index = 0; index < listSubjects.length; index++ ) {

		let textID = listSubjects[index].text;

		// NOTE: Each index of each subject in the various lists is currently different
		// Adjusting NIAK output to output dictionaries instead of lists would
		// remove the need for this work
		let indices = {

			t1: 	   getIndexOfKeyInList(textID, dataT1[0]),
			overlapT1: getIndexOfKeyInList(textID, dataOverlapT1[0]),
			bold: 	   getIndexOfKeyInList(textID, dataBold[0]),
			brain: 	   getIndexOfKeyInList(textID, dataBrain[0])
		};

		subjectTable[textID] = 
			new QCDash_Subject(listSubjects[index].id, 
							   textID,
							   dataT1[1][indices.t1],
							   dataOverlapT1[1][indices.overlapT1],
							   dataBold[1][indices.bold],
							   dataBrain[1][indices.brain],
							   indices);
	}

	return subjectTable;
}

function createRunDictionary(listRuns, dataIntra, dataFd, dataNbVol) {

	var runTable = {};

	// Create objects for each run
	for ( let index = 0; index < listRuns.length; index++ ) {

		let textID = listRuns[index].text;

		// NOTE: Each index of each run in the various lists is currently different
		// Adjusting NIAK output to output dictionaries instead of lists would
		// remove the need for this work
		let indices = {

			intra: getIndexOfKeyInList(textID, dataIntra[0]),
			fd:    getIndexOfKeyInList(textID, dataFD[0]),
			nbVol: getIndexOfKeyInList(textID, dataNbVol[0])
		};

		runTable[textID] = 
			new QCDash_Run(listRuns[index].id, 
						   textID,
						   dataIntra[1][indices.intra],
						   dataFD[1][indices.fd],
						   dataFD[2][indices.fd],
						   dataNbVol[1][indices.nbVol],
						   dataNbVol[2][indices.nbVol],
						   indices);

	}

	return runTable;
}

// Event functions

function sendChartToSelected(p_selectedWrapperID) {

}

// Graph objects

class QCDash_BarChart {

    constructor(p_data, p_divID, p_xLabel, p_yLabel, p_height, p_clickFn = null, p_chartTitle = "", p_chartNotes = "", p_subChart = false, p_extents = [0, 25]) {

        // Parameters
        this.m_data = p_data;
        this.m_divID = p_divID;
        this.m_labels = { x: p_xLabel, y: p_yLabel };
        this.m_dimensions = { width: window.innerWidth * 0.75 , height: p_height };
        this.m_clickFn = p_clickFn;
        this.m_chartTitle = p_chartTitle;
        this.m_chartNotes = p_chartNotes;
        this.m_extents = p_extents;

        // Reference to c3 chart
        this.m_chartRef = null;

        // Y-tick formatting
        this.m_yTickFormat = d3.format("0.2f");

        // Check for optional subchart flag
        let mySubChartOptions = null;
        let myExtents = null;
        this.m_hasSubChart = p_subChart;
        if ( p_subChart ) {

            mySubChartOptions = {

                axis: {

                    x: {

                        label: this.m_labels.x,
                        show: false,
                        type: "category",
                    },    
                    y: {

                        label: this.m_labels.y,
                        tick: { format: this.m_yTickFormat }                            
                    }
                },
                show: true
            };

            myExtents = this.m_extents;
        }

        // Set up chart fields
        this.m_chartFields = {

            axis: {        
                
                x: {

                    extent: myExtents,
                    label: this.m_labels.x,
                    show: false,
                    type: "category" 
                },    
                y: {

                    label: {
                        text: this.m_labels.y,
                        position: "outer-middle"
                    },
                    tick: { format: this.m_yTickFormat }
                }
            },

            // bar: {
            
            //     width: {
            //         ratio: 1.2
            //     }
            // },

            bindto: "#" + this.m_divID,

            color: {
                pattern: ["#00d1b2", "#ff3860"],
            },

            data: {
        
                columns: this.m_data,
                onclick: function(d, element) { },
                ondblclick: function (d) { this.m_clickFn(d, this.m_chartRef); }.bind(this),
                type: "bar",
                x: this.m_labels.x,
  				selection: {
    				enabled: true,
                    multiple: true
  				}
            },

            // selection: { enabled: true },

            size: this.m_dimensions,                   

            subchart: mySubChartOptions,

            zoom: { enabled: true },                    
        };
    }

    addClickEvent(p_titleID, p_stageID, p_notesID, p_selectedWrapperID) {

        // $(p_titleID.on("click", ))
    }

    generate() {

        // Generate chart and save a reference to the c3 object
        this.m_chartRef = c3.generate(this.m_chartFields);
    }

    focus(p_dataPointName) {

    	this.m_chartRef.focus(p_dataPointName);
    }
}

class QCDash_LineChart {

    constructor(p_data, p_divID, p_xLabel, p_yLabel, p_height, p_clickFn = null, p_chartTitle = "", p_chartNotes = "", p_extents = [0, 25]) {

        // Parameters
        this.m_data = p_data;
        this.m_divID = p_divID;
        this.m_labels = { x: p_xLabel, y: p_yLabel };
        this.m_dimensions = { height: p_height };
        this.m_chartTitle = p_chartTitle;
        this.m_chartNotes = p_chartNotes;        
        this.m_clickFn = p_clickFn;
        this.m_extents = p_extents;

        // Reference to c3 chart
        this.m_chartRef = null;

        // Y-tick formatting
        this.m_yTickFormat = d3.format("0.2f");

        // Set up chart fields
        this.m_chartFields = {

            axis: {        
                
                x: {

                    //extent: this.m_extents,
                    label: this.m_labels.x,
                    show: false,
                    //type: "category" 
                },    
                y: {

                    label: {
                        text: this.m_labels.y,
                        position: "outer-middle"
                    },
                    tick: { format: this.m_yTickFormat }
                }
            },                            

            bindto: "#" + this.m_divID,

            color: {
                pattern: ["#00d1b2", "#ff3860", "#3273DC"],
            },

            data: this.m_data,

            selection: { enabled: true },

            size: this.m_dimensions,                 

            // subchart: {

            //     axis: {

            //         x: {

            //             label: this.m_labels.x,
            //             show: false,
            //             type: "category",
            //         },    
            //         y: {

            //             label: this.m_labels.y,
            //             tick: { format: this.m_yTickFormat }                            
            //         }
            //     },
            //     show: true
            // },

            zoom: { enabled: true },                    
        };
    }

    generate() {

        // Generate chart and save a reference to the c3 object
        this.m_chartRef = c3.generate(this.m_chartFields);
    }

    focus(p_dataPointName) {
    	
    	this.m_chartRef.focus(p_dataPointName);
    }    
}

// class QCDash_StorageBased_TextArea {

//     constructor(p_textareaID, p_storageElement) {

//     }

//     load() {


//     }
// }

// class QCDash_StorageBased_Button {

//     constructor(p_buttonID, p_storageElement, p_statusField, p_statusText) {

//         this.m_buttonID = p_buttonID;
//         this.m_statusField = p_statusField;
//         this.m_statusText = p_statusText;
//         this.m_storageElement = p_storageElement;
//     }

//     load() {

//     }

//     setSubjectStatus(p_subject) {

//         // Update local data with this button's status for the current subject
//         if ( !this.m_storageElement.hasSubjectRecord(p_subject) ) {
            
//             this.m_storageElement.createSubjectRecord();
//         }
//         this.m_storageElement.m_localStorage.data.subjects[p_subject].status = this.m_status;

//         // Save local data to web storage
//         this.m_storageElement.saveLocalDataToStorage();
//     }   
// }

class QCDash_WebStorage_SubjectData {

    constructor(p_exportFilePrefix, p_defaultText, p_textAreaID, p_currentSubject, p_setSubjectCallback) {

        // Subject fields
        this.m_currentSubject = p_currentSubject;
        this.m_setSubjectCallback = p_setSubjectCallback;

        // File prefix (will also fill in as storage ID)
        this.m_exportFilePrefix = p_exportFilePrefix;

        // Get date for this webstorage instance
        let currentDate = new Date();
        let dateString = currentDate.getDate() + "-" + 
                         (currentDate.getMonth() + 1) + "-" + currentDate.getFullYear();
        let timeStamp = currentDate.getTime();

        // Local copy of web storage data
        this.m_localStorage = {

            // Prefix will be ID, loads last browser session
            // Single copy under prefix lowers amount of data stored in browser web storage
            id: this.m_exportFilePrefix, 
            data: { 
                
                date: dateString,
                timestamp: timeStamp,
                exportFile: {},
                lastSubject: { name: "", id: -1 },
                subjects: {}, 
            }
        };

        // Load up info from local storage if it exists, if not create it
        if ( null !== localStorage.getItem(this.m_localStorage.id) ) {

            this.saveStorageDataToLocal();
        } else {

            this.saveLocalDataToStorage();
        }          

        // Create export filename (requires m_localStorage to be set up first)
        this.m_exportFilename = this.generateNextDashboardFilename();

        // Comments fields
        this.m_defaultText = p_defaultText;
        this.m_textAreaID = p_textAreaID;
        this.m_textAreaRef = $("#" + this.m_textAreaID);    

        // Set the default comments for the textarea
        document.getElementById(this.m_textAreaID).defaultValue = this.m_defaultText;        
        
        // Check for current subject
        // If subject exists, comment box gets text from web session storage
        // Else, comment box gets default text
        if ( Object.keys(this.m_localStorage.data.subjects).indexOf(this.m_currentSubject) > -1 ) {

            this.m_textAreaRef.val(this.m_localStorage.data.subjects[this.m_currentSubject].comments);
        } else {

            this.m_textAreaRef.val(this.m_defaultText);
        }

        // Set focus/default text behavior
        this.m_textAreaRef.focus(function() {

            if ( this.defaultValue === this.value ) {
                this.value = "";
            }
        })
        .blur(function() {

            if ( "" === this.value ) {
                this.value = this.defaultValue;
            }
        })
        .bind("input propertychange", function() {

            this.saveCommentsToWebStorage();
        }.bind(this));        
    }

    importSubjectInfo(p_importJSON) {

        // Object fields
        

        this.m_currentSubject = p_importJSON.lastSubject.name;
        this.m_exportFilePrefix = p_importJSON.exportFile.prefix;

        // Get date for this webstorage instance
        let currentDate = new Date();
        let dateString = currentDate.getDate() + "-" + 
                         (currentDate.getMonth() + 1) + "-" + currentDate.getFullYear();
        let timeStamp = currentDate.getTime();


        this.m_localStorage = {

            // Prefix will be ID, loads last browser session
            // Single copy under prefix lowers amount of data stored in browser web storage
            id: this.m_exportFilePrefix, 
            data: { 
                
                date: dateString,
                timestamp: timeStamp,
                exportFile: {

                    fileNumber: 1,
                    currentDate: dateString,
                    prefix: this.m_exportFilePrefix
                },
                lastSubject: { name: p_importJSON.lastSubject.name, 
                               id: p_importJSON.lastSubject.id },
                subjects: p_importJSON.subjects
            }
        };

        this.m_exportFilename = this.generateNextDashboardFilename();

        // Generate a new export filename
        p_webStorageObject.m_exportFilename = p_webStorageObject.generateNextDashboardFilename();

        // Copy web storage data to local record
        p_webStorageObject.saveStorageDataToLocal();

    }

    // Get/Set

    get currentSubject() {

        return this.m_currentSubject;
    }

    get data() { 

        return localStorage.getItem(this.m_localStorage.id); 
    }      

    get exportFilename() {

        return this.m_exportFilename;
    }

    get exportFilePrefix() {

        return this.m_exportFilePrefix;
    }

    get lastSubjectID() {

        return this.m_localStorage.data.lastSubject.id;
    }

    get lastSubjectName() {

        return this.m_localStorage.data.lastSubject.name;
    }

    get storageID() {

        return this.m_localStorage.id;
    }

    get subjects() {

        return this.m_localStorage.data.subjects;
    }


    // Subject Methods

    createSubjectRecord(p_subject) {

        this.m_localStorage.data.subjects[p_subject] = { comments: "", status: "" };
    }

    hasLastSubjectRecord() {

        return ( "" != this.m_localStorage.data.lastSubject.name );
    }

    hasSubjectRecord(p_subject) {

        return ( Object.keys(this.m_localStorage.data.subjects).indexOf(p_subject) > -1 );
    }

    setSubject(p_newSubject, p_subjectListID) {

        // Set the new subject as current as last-viewed subject 
        // (the latter for subsequent browser sessions)
        this.m_currentSubject = p_newSubject;
        this.m_localStorage.data.lastSubject.name = p_newSubject;
        this.m_localStorage.data.lastSubject.id = p_subjectListID;

        // Save the last subject info to actual web storage
        this.saveLocalDataToStorage();
    }

    setToLastSubject() {

        // Look for last viewed subject (safeguard against setting to blank lastSubject)
        if ( this.hasLastSubjectRecord() ) {

            this.m_currentSubject = this.m_localStorage.data.lastSubject.name;
            var event = {
                params: {
                    data: {
                        id: this.m_localStorage.data.lastSubject.id
                    }
                }
            }; 
            this.m_setSubjectCallback(event);
        }   
    }


    // Status methods

    getSubjectStatus(p_subject) {

        if ( !this.hasSubjectRecord(p_subject) ) {

            this.createSubjectRecord(p_subject);
        }
        return this.m_localStorage.data.subjects[p_subject].status;
    }

    setSubjectStatus(p_subject, p_status) {

        // Create a local record for this subject if it does not already exist
        if ( !this.hasSubjectRecord(p_subject) ) {
            
            this.createSubjectRecord(p_subject);
        }

        // Update local record with pass/maybe/fail status for the current subject
        this.m_localStorage.data.subjects[p_subject].status = p_status;

        // Update actual browser web storage
        this.saveLocalDataToStorage();
    }


    // Comment methods

    restoreCommentsToTextArea() {

        if ( "" != this.m_localStorage.data.subjects[this.m_currentSubject].comments ) {
            
            this.m_textAreaRef.val(this.m_localStorage.data.subjects[this.m_currentSubject].comments);
        } else {

            this.m_textAreaRef.val(this.m_defaultText);
        }
    }

    saveCommentsToWebStorage() {

        // Update web storage field with comments for current subject
        this.setSubjectComments(this.m_currentSubject);

        // Update actual browser web storage
        this.saveLocalDataToStorage();
    }

    setSubjectComments(p_subject, p_comments) {

        // Update web storage field with comments for the current subject
        if ( !this.hasSubjectRecord(this.m_currentSubject) ) {
            
            this.createSubjectRecord();
        }
        this.m_localStorage.data.subjects[this.m_currentSubject].comments = this.m_textAreaRef.val();        
    }


    // Web storage methods

    saveStorageDataToLocal() {

        this.m_localStorage.data = JSON.parse(localStorage.getItem(this.m_localStorage.id));
    }

    saveLocalDataToStorage() {

        localStorage.setItem(this.m_localStorage.id, JSON.stringify(this.m_localStorage.data));
    } 


    // File methods

    generateNextDashboardFilename() {

        // Get current date
        let today = new Date();
        let currentDate = today.getFullYear().toString() + 
                          (today.getMonth() + 1).toString() + 
                          today.getDate().toString();

        // File counter starts at 1
        let fileNumber = 1;

        // Export file JSON (defaults at version 1, current date, given file prefix)
        let exportFileJSON = { "fileNumber": 1, "currentDate": currentDate, 
                               "prefix": this.m_exportFilePrefix };

        // Check for filename in web storage
        if ( null !== localStorage.getItem(this.m_localStorage.id) ) {

            let fullWebStorageJSON = JSON.parse(localStorage.getItem(this.m_localStorage.id));
            if ( fullWebStorageJSON.subjects.exportFile ) {

                exportFileJSON = JSON.parse(fullWebStorageJSON.subjects.exportFile);
                if ( currentDate == exportFileJSON.currentDate ) {

                    exportFileJSON.fileNumber = parseInt(exportFileJSON.fileNumber) + 1;
                    fileNumber = exportFileJSON.fileNumber;
                } else {

                    exportFileJSON.currentDate = currentDate;
                    exportFileJSON.fileNumber = 1;
                }
            }
        }

        // Update export JSON in local record
        this.m_localStorage.data.exportFile = exportFileJSON;

        // Update file info in web storage
        this.saveLocalDataToStorage();

        // Return constructed filename
        return this.m_exportFilePrefix + currentDate + "_" + fileNumber + ".json";
    }


    // Static Methods

    // Blob version of file creation/writing
    static exportFile(p_localStorageID, p_exportFilename) {

        let jsonString = JSON.stringify(localStorage.getItem(p_localStorageID));
        if ( null === jsonString ) {
            return;
        }

        var data = new Blob([localStorage.getItem(p_localStorageID)], 
                            { type: "text/plain;charset=utf-8" });
        var fileOfBlob = new File([data], p_exportFilename);

        // Returns a URL you can use as a href
        return window.URL.createObjectURL(fileOfBlob);
    }    
 
    static importFile(p_webStorageObject, p_filename, p_subjectList, p_uiList, p_setUIToSubjectCallback) {

        if ( null !== p_filename ) {

            var reader = new FileReader();
            reader.readAsText(p_filename, "UTF-8");
            reader.onload = function (p_event) {

                // $("#" + p_uiList.comments).val("Data file: " + p_event.target.result);

                try {

                    // Load local web storage object
                    let importedJSON = JSON.parse(p_event.target.result);
                    localStorage.setItem(importedJSON.exportFile.prefix, 
                                         p_event.target.result);

                    // New QCDash_WebStorage_SubjectData changeover function here
                    // This might do a few of the things below
                    importSubjectInfo(importedJSON);


                    // Copy web storage data to local record
                    p_webStorageObject.saveStorageDataToLocal();

                    // Generate a new export filename
                    p_webStorageObject.m_exportFilename = p_webStorageObject.generateNextDashboardFilename();

                    // Fill in the UI
                    p_setUIToSubjectCallback(p_webStorageObject, p_subjectList, p_uiList);
                    
                } catch ( error ) {

                    console.log("Error: " + error.message);
                    $("#" + p_uiList.comments).val("Error reading file...");                
                }
            }.bind(this);
            reader.onerror = function (p_event) {

                $("#" + p_uiList.comments).val("Error reading file...")
            }.bind(this);
        }
    }
}

// Code that should be run when deploying QC dashboard class objects

// Indicates filename when it is selected by file choice label
// From: https://tympanus.net/codrops/2015/09/15/styling-customizing-file-inputs-smart-way/
// var inputs = document.querySelectorAll( "[class^='inputfile']" );
var inputs = document.querySelectorAll( ".inputfile-import" );
Array.prototype.forEach.call(inputs, function(input) {
    
    var label    = input.nextElementSibling,
        labelVal = label.innerHTML;

    input.addEventListener("change", function(e) {
        
        var fileName = "";
        if ( this.files && this.files.length > 1 ) {
            fileName = ( this.getAttribute("data-multiple-caption") || "" ).replace( "{count}", this.files.length );
        } else {
            fileName = e.target.value.split( "\\" ).pop();
        }

        if ( fileName ) {
            label.querySelector("span").innerHTML = fileName;
        } else {
            label.innerHTML = labelVal;
        }
    });
});